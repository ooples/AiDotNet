using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Non-Attentive Tacotron: Tacotron 2 with its attention replaced by an explicit duration predictor and Gaussian
/// upsampling (supervised-duration variant).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including
/// Unsupervised Duration Modeling" (Shen et al., 2020), §3 and Appendix A (Table 6).</para>
/// <para>
/// The encoder embeds phonemes and applies three (dropout, batch normalization, convolution) layers and a
/// bidirectional LSTM with zoneout. A duration predictor (two bidirectional LSTMs and a projection) predicts each
/// token's duration in seconds; a range predictor (two bidirectional LSTMs on the encoder output and the durations, a
/// projection and SoftPlus) predicts σ. Gaussian upsampling (§3.1) spreads each encoder output over the frames with a
/// Gaussian centred on its segment, and a sinusoidal embedding of each frame's index within its token is concatenated.
/// The autoregressive decoder feeds the previous frame through a two-layer pre-net, concatenates the current upsampled
/// encoder output, runs two zoneout LSTMs, concatenates the upsampled output again and projects to r frames; a 5-layer
/// post-net adds a residual. Training uses target durations for upsampling and minimizes L1 + L2 on the spectrogram
/// before and after the post-net plus 2.0 × the squared duration error in seconds.
/// </para>
/// <para>Durations come from outside the model (supervised training), so a plain <c>Train(tokens, mel)</c> throws; pass a
/// <see cref="TtsTrainingSample{T}"/> with <see cref="TtsTrainingSample{T}.Durations"/> (in frames).</para>
/// <para><b>For Beginners:</b> Like Tacotron 2, but instead of letting attention guess where it is in the text, it is
/// told how long each sound lasts, so it cannot skip or repeat words.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling",
    "https://arxiv.org/abs/2010.04301",
    Year = 2020,
    Authors = "Shen et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-6, WeightDecay = 1e-6,
                LearningRate = 0.001, WarmupSteps = 4000,
                Schedule = LearningRateSchedulerType.Step, StepSize = 50000, DecayRate = 0.5,
                Source = "Shen et al. 2020, Appendix A, Table 6: Adam(0.9, 0.999, 1e-6) at a learning rate of 0.001, with a linear rampup over 4K steps then halving every 50K steps; L2 regularization 1e-6.")]
public partial class NonAttentiveTacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly NonAttentiveTacotronOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    private EmbeddingLayer<T>? _embedding;
    private readonly List<(DropoutLayer<T>? Dropout, BatchNormalizationLayer<T> Norm, Conv1DLayer<T> Conv)> _encoderConvs = new();
    private BidirectionalRecurrentLayer<T>? _encoderLstm;
    private BidirectionalRecurrentLayer<T>? _durationLstm1;
    private BidirectionalRecurrentLayer<T>? _durationLstm2;
    private DenseLayer<T>? _durationProjection;
    private BidirectionalRecurrentLayer<T>? _rangeLstm1;
    private BidirectionalRecurrentLayer<T>? _rangeLstm2;
    private DenseLayer<T>? _rangeProjection;
    private readonly List<LayerBase<T>> _prenet = new();
    private LSTMCellLayer<T>? _decoderLstm1;
    private LSTMCellLayer<T>? _decoderLstm2;
    private DenseLayer<T>? _frameProjection;
    private ConvBatchNormStackLayer<T>? _postnet;

    public override ModelOptions GetOptions() => _options;

    public NonAttentiveTacotron(NeuralNetworkArchitecture<T> architecture, string modelPath, NonAttentiveTacotronOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new NonAttentiveTacotronOptions();
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

    public NonAttentiveTacotron(
        NeuralNetworkArchitecture<T> architecture,
        NonAttentiveTacotronOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new NonAttentiveTacotronOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    /// <inheritdoc />
    protected override int TargetFftSize => _options.FftSize;

    /// <inheritdoc />
    protected override int TargetWindowSize => _options.WindowSize;

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
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EmbeddingDim);
        int previous = o.EmbeddingDim;
        foreach (int channels in o.EncoderConvChannels)
        {
            _encoderConvs.Add((o.DropoutRate > 0 ? new DropoutLayer<T>(o.DropoutRate) : null,
                new BatchNormalizationLayer<T>(previous, epsilon: 1e-3, momentum: o.BatchNormDecay),
                new Conv1DLayer<T>(inputChannels: previous, outputChannels: channels, kernelSize: o.EncoderKernelSize)));
            previous = channels;
        }
        _encoderLstm = new BidirectionalRecurrentLayer<T>(previous, o.EncoderLstmDim, RecurrentCellType.Lstm);
        int encoderWidth = 2 * o.EncoderLstmDim;
        _durationLstm1 = new BidirectionalRecurrentLayer<T>(encoderWidth, o.DurationLstmDim, RecurrentCellType.Lstm);
        _durationLstm2 = new BidirectionalRecurrentLayer<T>(2 * o.DurationLstmDim, o.DurationLstmDim, RecurrentCellType.Lstm);
        _durationProjection = new DenseLayer<T>(1, identity);
        _rangeLstm1 = new BidirectionalRecurrentLayer<T>(encoderWidth + 1, o.RangeLstmDim, RecurrentCellType.Lstm);
        _rangeLstm2 = new BidirectionalRecurrentLayer<T>(2 * o.RangeLstmDim, o.RangeLstmDim, RecurrentCellType.Lstm);
        _rangeProjection = new DenseLayer<T>(1, identity);
        foreach (int size in o.PrenetSizes)
        {
            _prenet.Add(new DenseLayer<T>(size, new ReLUActivation<T>() as IActivationFunction<T>));
            _prenet.Add(new DropoutLayer<T>(o.PrenetDropout));
        }
        int upsampledWidth = encoderWidth + o.PositionalEmbeddingDim;
        _decoderLstm1 = new LSTMCellLayer<T>(o.PrenetSizes[^1] + upsampledWidth, o.DecoderLstmDim, o.LstmCellClip);
        _decoderLstm2 = new LSTMCellLayer<T>(o.DecoderLstmDim, o.DecoderLstmDim, o.LstmCellClip);
        _frameProjection = new DenseLayer<T>(o.MelChannels * o.OutputsPerStep, identity);
        var postnetChannels = Enumerable.Repeat(o.PostnetDim, o.PostnetLayers - 1).Append(o.MelChannels).ToArray();
        _postnet = new ConvBatchNormStackLayer<T>(o.MelChannels, postnetChannels, o.PostnetKernelSize, useTanh: true,
            linearLast: true, o.PostnetDropout);

        var encoder = new List<ILayer<T>> { _embedding };
        foreach (var (dropout, norm, conv) in _encoderConvs)
        {
            if (dropout is not null) encoder.Add(dropout);
            encoder.Add(norm);
            encoder.Add(conv);
        }
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(new LayerBase<T>[] { _encoderLstm, _durationLstm1, _durationLstm2, _rangeLstm1, _rangeLstm2 });
        ComponentLayers.Add(_durationProjection);
        ComponentLayers.Add(_rangeProjection);
        ComponentLayers.AddRange(_prenet);
        ComponentLayers.AddRange(new LayerBase<T>[] { _decoderLstm1, _decoderLstm2, _frameProjection, _postnet });
    }

    private bool HasPaperLayers => _embedding is not null;

    /// <summary>The encoder output <c>H</c>, <c>[tokens, 2 · LSTM]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> tokens)
    {
        var x = _embedding!.Forward(tokens);
        int length = x.Shape[0];
        foreach (var (dropout, norm, conv) in _encoderConvs)
        {
            if (dropout is not null) x = dropout.Forward(x);
            x = norm.Forward(x);
            var y = conv.Forward(Engine.Reshape(Engine.TensorTranspose(x), new[] { 1, x.Shape[1], length }));
            x = Engine.TensorTranspose(Engine.Reshape(y, new[] { y.Shape[1], length }));
        }
        return _encoderLstm!.Forward(x);
    }

    private Tensor<T> PredictDurationsSeconds(Tensor<T> encoded)
        => Engine.Reshape(_durationProjection!.Forward(_durationLstm2!.Forward(_durationLstm1!.Forward(encoded))), new[] { encoded.Shape[0] });

    private Tensor<T> PredictRanges(Tensor<T> encoded, Tensor<T> durationsSeconds)
    {
        var input = Engine.TensorConcatenate(new[] { encoded, Engine.Reshape(durationsSeconds, new[] { encoded.Shape[0], 1 }) }, 1);
        var projected = _rangeProjection!.Forward(_rangeLstm2!.Forward(_rangeLstm1!.Forward(input)));
        // SoftPlus, written stably: log(1 + e^x) = max(x, 0) + log(1 + e^-|x|).
        var softplus = Engine.TensorAdd(Engine.ReLU(projected),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(projected))), NumOps.One)));
        return Engine.Reshape(softplus, new[] { encoded.Shape[0] });
    }

    /// <summary>
    /// Gaussian upsampling (§3.1, Eq. 4–6): <c>u_t = Σ_i w_ti h_i</c> with <c>w_ti ∝ N(t; c_i, σ_i²)</c> and
    /// <c>c_i = d_i / 2 + Σ_{j&lt;i} d_j</c>, followed by the concatenated positional embedding of each frame's index within
    /// its token. Returns <c>[frames, encoder + positional]</c>.
    /// </summary>
    private Tensor<T> GaussianUpsample(Tensor<T> encoded, int[] durations, Tensor<T> sigma)
    {
        int tokens = durations.Length, frames = Math.Max(1, durations.Sum());
        var centres = new Tensor<T>(new[] { 1, tokens });
        double cumulative = 0;
        for (int i = 0; i < tokens; i++)
        {
            centres[0, i] = NumOps.FromDouble(durations[i] / 2.0 + cumulative);
            cumulative += durations[i];
        }
        var t = new Tensor<T>(new[] { frames, 1 });
        for (int f = 0; f < frames; f++) t[f, 0] = NumOps.FromDouble(f);
        var diff = Engine.TensorSubtract(Engine.TensorTile(t, new[] { 1, tokens }), Engine.TensorTile(centres, new[] { frames, 1 }));
        var sig = Engine.TensorTile(Engine.Reshape(sigma, new[] { 1, tokens }), new[] { frames, 1 });
        // log N(t; c, σ²) = −(t − c)² / (2σ²) − log σ − ½ log 2π; normalizing over tokens is a softmax of the log densities.
        var logDensity = Engine.TensorSubtract(
            Engine.TensorNegate(Engine.TensorDivide(Engine.TensorMultiply(diff, diff), Engine.TensorMultiplyScalar(Engine.TensorMultiply(sig, sig), NumOps.FromDouble(2.0)))),
            Engine.TensorLog(sig));
        var weights = Engine.TensorSoftmax(logDensity, axis: 1);                                  // [frames, tokens]
        var upsampled = Engine.TensorMatMul(weights, encoded);

        int dim = _options.PositionalEmbeddingDim;
        var positions = new Tensor<T>(new[] { frames, dim });
        int frame = 0;
        foreach (int d in durations)
            for (int k = 1; k <= d && frame < frames; k++, frame++)
                for (int c = 0; c < dim; c++)
                {
                    double angle = k / Math.Pow(10000.0, (c - c % 2) / (double)dim);
                    positions[frame, c] = NumOps.FromDouble(c % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                }
        return Engine.TensorConcatenate(new[] { upsampled, positions }, 1);
    }

    /// <summary>
    /// The autoregressive decoder for <paramref name="steps"/> steps of r frames: pre-net on the previous frame (the
    /// all-zero frame first), concatenated with the upsampled encoder output of the step's first frame, two zoneout LSTMs,
    /// concatenated with it again and projected to r frames. With <paramref name="teacher"/> each step reads the last
    /// ground-truth frame of the previous group.
    /// </summary>
    private Tensor<T> Decode(Tensor<T> upsampled, int steps, Tensor<T>? teacher)
    {
        var o = _options;
        int r = o.OutputsPerStep, frames = upsampled.Shape[0];
        var state1 = new Tensor<T>(new[] { 1, 2 * o.DecoderLstmDim });
        var state2 = new Tensor<T>(new[] { 1, 2 * o.DecoderLstmDim });
        var previous = new Tensor<T>(new[] { 1, o.MelChannels });
        var groups = new List<Tensor<T>>(steps);
        for (int s = 0; s < steps; s++)
        {
            var x = previous;
            foreach (var layer in _prenet) x = layer.Forward(x);
            var u = Engine.TensorSlice(upsampled, new[] { Math.Min(s * r, frames - 1), 0 }, new[] { 1, upsampled.Shape[1] });
            state1 = Zoneout(state1, _decoderLstm1!.Forward(Engine.TensorConcatenate(new[] { x, u }, 1), state1));
            var h1 = _decoderLstm1.SplitState(state1).Hidden;
            state2 = Zoneout(state2, _decoderLstm2!.Forward(h1, state2));
            var h2 = _decoderLstm2.SplitState(state2).Hidden;
            var group = Engine.Reshape(_frameProjection!.Forward(Engine.TensorConcatenate(new[] { h2, u }, 1)), new[] { r, o.MelChannels });
            groups.Add(group);
            previous = teacher is not null
                ? Engine.TensorSlice(teacher, new[] { s * r + r - 1, 0 }, new[] { 1, o.MelChannels })
                : new Tensor<T>(new[] { 1, o.MelChannels }, Engine.TensorSlice(group, new[] { r - 1, 0 }, new[] { 1, o.MelChannels }).ToVector());
        }
        return groups.Count == 1 ? groups[0] : Engine.TensorConcatenate(groups.ToArray(), 0);
    }

    // Zoneout (Krueger et al. 2017): in training each unit keeps its previous value with probability p; at inference the
    // expectation p · previous + (1 − p) · new is used.
    private Tensor<T> Zoneout(Tensor<T> previous, Tensor<T> next)
    {
        double p = _options.ZoneoutProbability;
        if (p <= 0) return next;
        if (!IsTrainingMode)
            return Engine.TensorAdd(Engine.TensorMultiplyScalar(previous, NumOps.FromDouble(p)), Engine.TensorMultiplyScalar(next, NumOps.FromDouble(1 - p)));
        var mask = new Tensor<T>(next._shape);
        for (int i = 0; i < mask.Length; i++) mask[i] = _zoneoutRandom.NextDouble() < p ? NumOps.One : NumOps.Zero;
        var keep = Engine.TensorAddScalar(Engine.TensorNegate(mask), NumOps.One);
        return Engine.TensorAdd(Engine.TensorMultiply(mask, previous), Engine.TensorMultiply(keep, next));
    }

    private readonly Random _zoneoutRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);

    private int[] FramesFromSeconds(Tensor<T> seconds)
    {
        var durations = new int[seconds.Length];
        double framesPerSecond = (double)_options.SampleRate / _options.HopSize;
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)Math.Max(0, Math.Round(NumOps.ToDouble(seconds[i]) * framesPerSecond * _options.DurationScale));
        if (durations.Sum() == 0) durations[durations.Length - 1] = 1;
        return durations;
    }

    /// <inheritdoc />
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
        var encoded = Encode(tokens);
        var seconds = PredictDurationsSeconds(encoded);
        var durations = FramesFromSeconds(seconds);
        var upsampled = GaussianUpsample(encoded, durations, PredictRanges(encoded, seconds));
        int r = _options.OutputsPerStep, frames = durations.Sum(), steps = (frames + r - 1) / r;
        var before = Decode(upsampled, steps, null);
        var after = Engine.TensorAdd(before, _postnet!.Forward(before));
        return Engine.TensorSlice(after, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
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
    /// Eq. 1–3: <c>L = L_spec + λ_dur L_dur</c>, with <c>L_spec</c> the mean of L1 + L2 errors before and after the
    /// post-net and <c>L_dur</c> the mean squared duration error in seconds; target durations drive the upsampling.
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(NonAttentiveTacotron<T>)} trains on target durations; set {nameof(sample.Durations)}.", nameof(sample));
        var targets = DeriveAcousticTargets(sample);
        int frames = targets.MelFrames, r = _options.OutputsPerStep;
        if (durations.Sum() != frames)
            throw new ArgumentException($"The durations sum to {durations.Sum()} frames but the mel spectrogram has {frames}.", nameof(sample));
        int steps = (frames + r - 1) / r;
        var padded = new Tensor<T>(new[] { steps * r, _options.MelChannels });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < _options.MelChannels; c++) padded[f, c] = targets.Mel[f, c];
        double secondsPerFrame = (double)_options.HopSize / _options.SampleRate;
        var targetSeconds = new Tensor<T>(new[] { durations.Length });
        for (int i = 0; i < durations.Length; i++) targetSeconds[i] = NumOps.FromDouble(durations[i] * secondsPerFrame);
        // Upsampling uses the target durations; the target frame count is padded to whole decoder steps.
        var upsamplingDurations = (int[])durations.Clone();
        upsamplingDurations[^1] += steps * r - frames;

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            var encoded = Encode(tokens);
            var predictedSeconds = PredictDurationsSeconds(encoded);
            var upsampled = GaussianUpsample(encoded, upsamplingDurations, PredictRanges(encoded, targetSeconds));
            var before = Decode(upsampled, steps, target);
            var after = Engine.TensorAdd(before, _postnet!.Forward(before));
            var spectrogram = Engine.TensorAdd(SpectrogramError(before, target), SpectrogramError(after, target));
            var durationError = Engine.TensorSubtract(predictedSeconds, targetSeconds);
            var durationLoss = Engine.ReduceMean(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false);
            return Engine.TensorAdd(spectrogram, Engine.TensorMultiplyScalar(durationLoss, NumOps.FromDouble(_options.DurationLossWeight)));
        }

        return (padded, Objective);
    }

    // (1 / TK) Σ (|e| + e²)
    private Tensor<T> SpectrogramError(Tensor<T> prediction, Tensor<T> target)
    {
        var e = Engine.TensorSubtract(prediction, target);
        return Engine.ReduceMean(Engine.TensorAdd(Engine.TensorAbs(e), Engine.TensorMultiply(e, e)), new[] { 0, 1 }, keepDims: false);
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "NonAttentiveTacotron-Native" : "NonAttentiveTacotron-ONNX",
            Description = "Non-Attentive Tacotron (Shen et al., 2020)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.EncoderConvChannels.Length + _options.PostnetLayers,
        };
        m.AdditionalInfo["Architecture"] = "NonAttentiveTacotron";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(NonAttentiveTacotron<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
