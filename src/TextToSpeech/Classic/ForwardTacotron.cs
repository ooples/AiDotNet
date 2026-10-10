using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// ForwardTacotron: a non-autoregressive Tacotron that predicts phoneme durations, pitch and energy and generates the
/// whole mel spectrogram in one pass.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> ForwardTacotron (Axel Springer AI, as-ideas/ForwardTacotron), a model published as a reference
/// implementation rather than a paper; it is inspired by FastSpeech and built from Tacotron components. The
/// autoregressive Non-Attentive Tacotron (Shen et al. 2020) is a different model.</para>
/// <para>
/// Three series predictors (phoneme embedding, three kernel-5 convolutions with ReLU and batch normalization, a
/// bidirectional GRU, a linear projection) predict each phoneme's duration in frames and its normalized pitch and energy.
/// The mel path embeds the phonemes, runs a CBHG pre-net (<see cref="CbhgVariant.ForwardTacotron"/>), adds the pitch and
/// energy series through kernel-3 convolutions, expands by the durations, and decodes with a bidirectional LSTM, a linear
/// projection and a CBHG post-net with a bias-free projection. Training uses the ground-truth durations (from a
/// Tacotron's attention), phoneme-level pitch (mean voiced WORLD F0 in 30–600 Hz) and energy (mean L2 norm of exp(mel)),
/// with L1 losses on mel, post-net mel and 0.1 × each series.
/// </para>
/// <para>Durations come from a forced alignment outside the model, so a plain <c>Train(tokens, mel)</c> throws; pass a
/// <see cref="TtsTrainingSample{T}"/> with <see cref="TtsTrainingSample{T}.Durations"/>.</para>
/// <para><b>For Beginners:</b> ForwardTacotron decides how long, how high and how loud each sound is, then paints the
/// whole spectrogram at once, so it never skips or repeats words.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "ForwardTacotron: Generating Speech in a Single Forward Pass",
    "https://github.com/as-ideas/ForwardTacotron",
    Year = 2020,
    Authors = "Schäfer, Axel Springer AI"
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, MaxGradientNorm = 1.0,
                Source = "as-ideas/ForwardTacotron configs/singlespeaker.yaml: Adam (PyTorch defaults) with the progressive schedule 5e-5 for 150k steps then 1e-5, clip_grad_norm 1.0.")]
public partial class ForwardTacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly ForwardTacotronOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    private EmbeddingLayer<T>? _embedding;
    private SeriesPredictor? _durationPredictor;
    private SeriesPredictor? _pitchPredictor;
    private SeriesPredictor? _energyPredictor;
    private CbhgLayer<T>? _prenet;
    private WeightNormlessConv? _pitchProjection;
    private WeightNormlessConv? _energyProjection;
    private BidirectionalRecurrentLayer<T>? _lstm;
    private DenseLayer<T>? _melProjection;
    private CbhgLayer<T>? _postnet;
    private BiasFreeLinearLayer<T>? _postProjection;

    public override ModelOptions GetOptions() => _options;

    public ForwardTacotron(NeuralNetworkArchitecture<T> architecture, string modelPath, ForwardTacotronOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ForwardTacotronOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public ForwardTacotron(
        NeuralNetworkArchitecture<T> architecture,
        ForwardTacotronOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ForwardTacotronOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        // Adam with PyTorch's defaults; the progressive schedule sets the rate (5e-5 for 150k steps, then 1e-5).
        double first = _options.LearningRate, second = _options.SecondStageLearningRate;
        int switchStep = _options.SecondStageStartStep;
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = first,
                    LearningRateScheduler = new LambdaLRScheduler(first, step => step < switchStep ? 1.0 : second / first),
                    UseAdaptiveBetas = false,
                }));
        MaxGradNorm = NumOps.FromDouble(_options.GradientClipNorm);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a log-mel spectrogram from text (pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>Durations come from a Tacotron's attention (the repository's duration extraction), outside this model.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EmbeddingDim);
        _durationPredictor = new SeriesPredictor(this, o.VocabSize, o.SeriesEmbeddingDim, o.DurationConvDim, o.DurationRnnDim, o.DurationDropout);
        _pitchPredictor = new SeriesPredictor(this, o.VocabSize, o.SeriesEmbeddingDim, o.PitchConvDim, o.PitchRnnDim, o.PitchDropout);
        _energyPredictor = new SeriesPredictor(this, o.VocabSize, o.SeriesEmbeddingDim, o.EnergyConvDim, o.EnergyRnnDim, o.EnergyDropout);
        _prenet = new CbhgLayer<T>(o.EmbeddingDim, o.PrenetBankSize, o.PrenetDim, new[] { o.PrenetDim, o.EmbeddingDim },
            o.PrenetDim, o.PrenetDim, CbhgVariant.ForwardTacotron, o.PrenetDropout);
        _pitchProjection = new WeightNormlessConv(this, 1, 2 * o.PrenetDim, 3);
        _energyProjection = new WeightNormlessConv(this, 1, 2 * o.PrenetDim, 3);
        _lstm = new BidirectionalRecurrentLayer<T>(2 * o.PrenetDim, o.RnnDim, RecurrentCellType.Lstm);
        _melProjection = new DenseLayer<T>(o.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _postnet = new CbhgLayer<T>(o.MelChannels, o.PostnetBankSize, o.PostnetChannels, new[] { o.PostnetChannels, o.MelChannels },
            o.PostnetChannels, o.PostnetChannels, CbhgVariant.ForwardTacotron, o.PostnetDropout);
        _postProjection = new BiasFreeLinearLayer<T>(2 * o.PostnetChannels, o.MelChannels);

        AddEncoderDecoderLayers(new List<ILayer<T>> { _embedding, _prenet }, Array.Empty<ILayer<T>>());
        foreach (var predictor in new[] { _durationPredictor, _pitchPredictor, _energyPredictor })
            ComponentLayers.AddRange(predictor.Layers);
        ComponentLayers.Add(_pitchProjection.Layer);
        ComponentLayers.Add(_energyProjection.Layer);
        ComponentLayers.AddRange(new LayerBase<T>[] { _lstm, _melProjection, _postnet, _postProjection });
    }

    private bool HasPaperLayers => _embedding is not null;

    /// <summary>
    /// The mel path (reference <c>_generate_mel</c>): embedding, CBHG pre-net, phoneme-level pitch and energy projected
    /// by a kernel-3 convolution and added, expansion by the durations, a bidirectional LSTM, a linear projection to mel,
    /// and the CBHG post-net with a bias-free projection (which replaces, rather than adds to, the mel).
    /// </summary>
    private (Tensor<T> Mel, Tensor<T> MelPost) Generate(Tensor<T> tokens, Tensor<T> pitch, Tensor<T> energy, int[] durations)
    {
        var x = _prenet!.Forward(_embedding!.Forward(tokens));                                       // [S, 2 * prenet]
        x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(_pitchProjection!.Forward(pitch), NumOps.FromDouble(_options.PitchStrength)));
        x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(_energyProjection!.Forward(energy), NumOps.FromDouble(_options.EnergyStrength)));
        x = LengthRegulator.Expand(x, durations);
        var mel = _melProjection!.Forward(_lstm!.Forward(x));
        var melPost = _postProjection!.Forward(_postnet!.Forward(mel));
        return (mel, melPost);
    }

    /// <summary>Durations used at inference: the prediction rounded to whole frames (negative to zero), or two frames per
    /// phoneme if every phoneme would vanish (reference <c>generate</c> and <c>LengthRegulator</c>).</summary>
    private int[] RoundedDurations(Tensor<T> predicted)
    {
        var durations = new int[predicted.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)Math.Floor(Math.Max(0.0, NumOps.ToDouble(predicted[i])) * _options.DurationScale + 0.5);
        if (durations.Sum() <= 0)
            for (int i = 0; i < durations.Length; i++) durations[i] = 2;
        return durations;
    }

    /// <inheritdoc />
    /// <remarks>Returns the post-net mel spectrogram.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        if (input.Rank == 1)
            return SynthesizeOne(input);
        if (input.Rank != 2)
            throw new ArgumentException($"Expected phonemes [phonemes] or [batch, phonemes], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], tokens = input.Shape[1];
        var outputs = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var row = new Tensor<T>(new[] { tokens });
            for (int i = 0; i < tokens; i++) row[i] = input[b, i];
            outputs.Add(SynthesizeOne(row));
        }
        int longest = outputs.Max(o => o.Shape[0]);
        var result = new Tensor<T>(new[] { batch, longest, _options.MelChannels });
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < outputs[b].Shape[0]; f++)
                for (int m = 0; m < _options.MelChannels; m++) result[b, f, m] = outputs[b][f, m];
        return result;
    }

    private Tensor<T> SynthesizeOne(Tensor<T> tokens)
    {
        using var _ = new NoGradScope<T>();
        var durations = RoundedDurations(_durationPredictor!.Forward(tokens));
        var pitch = _pitchPredictor!.Forward(tokens);
        var energy = _energyPredictor!.Forward(tokens);
        return Generate(tokens, pitch, energy, durations).MelPost;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var (mel, objective) = BuildObjective(sample);
        return TrainWithCustomObjective(sample.Tokens, mel, objective, _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (mel, objective) = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The reference training objective (<c>forward_trainer.py</c>): L1 on the mel and on the post-net mel, plus 0.1 ×
    /// L1 on the durations (frames), the phoneme-level pitch and the phoneme-level energy, with the ground-truth
    /// durations, pitch and energy driving the mel path.
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The objective needs the reference architecture; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(ForwardTacotron<T>)} trains on durations extracted from a Tacotron's attention; set {nameof(sample.Durations)}.",
            nameof(sample));
        var targets = DeriveAcousticTargets(sample);
        var mel = targets.Mel;
        if (durations.Sum() != targets.MelFrames)
            throw new ArgumentException($"The durations sum to {durations.Sum()} frames but the mel spectrogram has {targets.MelFrames}.", nameof(sample));
        var framePitch = targets.Pitch ?? throw new ArgumentException(
            $"{nameof(ForwardTacotron<T>)} conditions on phoneme-level pitch: supply the recording or Pitch.", nameof(sample));

        var (pitch, energy) = PhonemeLevelTargets(framePitch, mel, durations);
        var durationTarget = new Tensor<T>(new[] { durations.Length });
        for (int i = 0; i < durations.Length; i++) durationTarget[i] = NumOps.FromDouble(durations[i]);

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            var (predictedMel, predictedPost) = Generate(tokens, pitch, energy, durations);
            var loss = Engine.TensorAdd(MeanAbsolute(predictedMel, target), MeanAbsolute(predictedPost, target));
            var factor = NumOps.FromDouble(_options.VarianceLossFactor);
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanAbsolute(_durationPredictor!.Forward(tokens), durationTarget), factor));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanAbsolute(_pitchPredictor!.Forward(tokens), pitch), factor));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanAbsolute(_energyPredictor!.Forward(tokens), energy), factor));
            return loss;
        }

        return (mel, Objective);
    }

    /// <summary>
    /// Phoneme-level pitch (the mean of a phoneme's voiced F0 values within [PitchMinHz, PitchMaxHz], 0 if none) and
    /// energy (the mean over its frames of the L2 norm of exp(mel)), normalized by the dataset statistics in the options
    /// with unvoiced phonemes kept at 0 (reference <c>extract_pitch_energy</c>, <c>normalize_values</c>).
    /// </summary>
    private (Tensor<T> Pitch, Tensor<T> Energy) PhonemeLevelTargets(double[] framePitch, Tensor<T> mel, int[] durations)
    {
        int tokens = durations.Length, channels = mel.Shape[1];
        var pitch = new Tensor<T>(new[] { tokens });
        var energy = new Tensor<T>(new[] { tokens });
        int start = 0;
        for (int p = 0; p < tokens; p++)
        {
            double pitchSum = 0, energySum = 0;
            int voiced = 0;
            for (int f = start; f < start + durations[p]; f++)
            {
                double f0 = framePitch[f];
                if (f0 != 0 && f0 >= _options.PitchMinHz && f0 <= _options.PitchMaxHz)
                {
                    pitchSum += f0;
                    voiced++;
                }
                double norm = 0;
                for (int c = 0; c < channels; c++)
                {
                    double v = Math.Exp(NumOps.ToDouble(mel[f, c]));
                    norm += v * v;
                }
                energySum += Math.Sqrt(norm);
            }
            double phonemePitch = voiced > 0 ? pitchSum / voiced : 0.0;
            double phonemeEnergy = durations[p] > 0 ? energySum / durations[p] : 0.0;
            pitch[p] = NumOps.FromDouble(phonemePitch == 0.0 ? 0.0 : (phonemePitch - _options.PitchMean) / _options.PitchStd);
            energy[p] = NumOps.FromDouble(phonemeEnergy == 0.0 ? 0.0 : (phonemeEnergy - _options.EnergyMean) / _options.EnergyStd);
            start += durations[p];
        }
        return (pitch, energy);
    }

    private Tensor<T> MeanAbsolute(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorAbs(Engine.TensorSubtract(prediction, Engine.Reshape(target, prediction._shape)));
        return Engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), keepDims: false);
    }

    /// <summary>A 1-D convolution of a per-phoneme scalar series to <c>[phonemes, channels]</c> ("same" padding).</summary>
    private sealed class WeightNormlessConv
    {
        private readonly ForwardTacotron<T> _owner;
        public WeightNormlessConv(ForwardTacotron<T> owner, int input, int output, int kernel)
        {
            _owner = owner;
            Layer = new Conv1DLayer<T>(inputChannels: input, outputChannels: output, kernelSize: kernel);
        }

        public Conv1DLayer<T> Layer { get; }

        // [phonemes] -> [1, 1, phonemes] -> [1, output, phonemes] -> [phonemes, output]
        public Tensor<T> Forward(Tensor<T> series)
        {
            var engine = _owner.Engine;
            int length = series.Length;
            var y = Layer.Forward(engine.Reshape(series, new[] { 1, 1, length }));
            return engine.TensorTranspose(engine.Reshape(y, new[] { y.Shape[1], length }));
        }
    }

    /// <summary>
    /// The reference <c>SeriesPredictor</c>: its own phoneme embedding, three kernel-5 convolutions (no bias) each
    /// followed by ReLU, batch normalization and dropout, a bidirectional GRU, and a linear projection to one value per
    /// phoneme.
    /// </summary>
    private sealed class SeriesPredictor
    {
        private readonly ForwardTacotron<T> _owner;
        private readonly EmbeddingLayer<T> _embedding;
        private readonly List<(Conv1DLayer<T> Conv, BatchNormalizationLayer<T> Norm)> _convs = new();
        private readonly DropoutLayer<T>? _dropout;
        private readonly BidirectionalRecurrentLayer<T> _gru;
        private readonly DenseLayer<T> _projection;

        public SeriesPredictor(ForwardTacotron<T> owner, int vocab, int embedding, int conv, int rnn, double dropout)
        {
            _owner = owner;
            _embedding = new EmbeddingLayer<T>(vocab, embedding);
            int previous = embedding;
            for (int i = 0; i < 3; i++)
            {
                _convs.Add((new Conv1DLayer<T>(inputChannels: previous, outputChannels: conv, kernelSize: 5,
                    activation: new ReLUActivation<T>()), new BatchNormalizationLayer<T>(conv, epsilon: 1e-5, momentum: 0.9)));
                previous = conv;
            }
            _dropout = dropout > 0 ? new DropoutLayer<T>(dropout) : null;
            _gru = new BidirectionalRecurrentLayer<T>(conv, rnn, RecurrentCellType.Gru);
            _projection = new DenseLayer<T>(1, new IdentityActivation<T>() as IActivationFunction<T>);
        }

        public IEnumerable<LayerBase<T>> Layers
        {
            get
            {
                yield return _embedding;
                foreach (var (conv, norm) in _convs)
                {
                    yield return conv;
                    yield return norm;
                }
                if (_dropout is not null) yield return _dropout;
                yield return _gru;
                yield return _projection;
            }
        }

        public Tensor<T> Forward(Tensor<T> tokens)
        {
            var engine = _owner.Engine;
            var x = _embedding.Forward(tokens);                                       // [S, E]
            int length = x.Shape[0];
            foreach (var (conv, norm) in _convs)
            {
                var channelsFirst = engine.Reshape(engine.TensorTranspose(x), new[] { 1, x.Shape[1], length });
                var y = conv.Forward(channelsFirst);
                x = engine.TensorTranspose(engine.Reshape(y, new[] { y.Shape[1], length }));
                x = norm.Forward(x);
                if (_dropout is not null) x = _dropout.Forward(x);
            }
            return engine.Reshape(_projection.Forward(_gru.Forward(x)), new[] { length });
        }
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ForwardTacotron-Native" : "ForwardTacotron-ONNX",
            Description = "ForwardTacotron: non-autoregressive Tacotron with duration, pitch and energy predictors (as-ideas)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.PrenetBankSize + _options.PostnetBankSize,
        };
        m.AdditionalInfo["Architecture"] = "ForwardTacotron";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ForwardTacotron<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
