using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// SpeedySpeech: a fully convolutional non-autoregressive text-to-speech student network trained on durations from a
/// teacher's attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "SpeedySpeech: Efficient Neural Speech Synthesis" (Vainer &amp; Dušek, Interspeech 2020), and
/// the authors' implementation (github.com/janvainer/speedyspeech) for what the paper leaves unstated.</para>
/// <para>
/// The student (§3.2, Fig. 3) has three convolutional modules built from <see cref="DilatedResidualConvBlock{T}"/>:
/// a phoneme encoder (embedding, a fully connected layer with ReLU, 13 two-convolution residual blocks with dilations
/// 1, 2, 4 repeated, a long residual connection, and a normalized output projection), a duration predictor reading the
/// encodings with their gradient detached ("we detach gradient flow from the duration predictor to the encoder"), and
/// a decoder (17 blocks with dilations 1, 2, 4, 8 repeated) that turns the expanded encodings, plus a positional
/// encoding restarted at every phoneme, into a log-mel spectrogram. Training uses teacher durations for the expansion
/// and minimizes MAE + (1 − SSIM) on the standardized log-mel spectrogram plus the Huber loss of the log durations.
/// </para>
/// <para>
/// Training data: durations come from the teacher network (§3.1), so a plain <c>Train(tokens, mel)</c> throws; pass a
/// <see cref="TtsTrainingSample{T}"/> with <see cref="TtsTrainingSample{T}.Durations"/>.
/// </para>
/// <para><b>For Beginners:</b> SpeedySpeech decides how long each phoneme lasts, stretches the phonemes to that many
/// frames, and paints the whole spectrogram at once with stacks of convolutions — no attention, so it is fast.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SpeedySpeech: Efficient Neural Speech Synthesis",
    "https://arxiv.org/abs/2008.03802",
    Year = 2020,
    Authors = "Vainer and Dušek"
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.002, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.ReduceOnPlateau,
                Source = "Vainer and Dusek 2020, Sec. 4: the Adam optimizer with its default "
                        + "parameters, gradient clipping at 1, and a base learning rate of 0.002. The "
                        + "paper tried inverse-square-root decay and reduce-on-plateau and settled on "
                        + "reduce-on-plateau.")]
public partial class SpeedySpeech<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly SpeedySpeechOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Encoder: Embedding -> FC + ReLU (prenet) -> residual blocks -> FC (+ prenet output) -> ReLU -> BN -> FC.
    private EmbeddingLayer<T>? _embedding;
    private DenseLayer<T>? _prenet;
    private readonly List<DilatedResidualConvBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _encoderPost1;
    private BatchNormalizationLayer<T>? _encoderPostNorm;
    private DenseLayer<T>? _encoderPost2;

    // Decoder: residual blocks -> FC (+ input) -> residual block -> FC to mel.
    private readonly List<DilatedResidualConvBlock<T>> _decoderBlocks = new();
    private DenseLayer<T>? _decoderPost1;
    private DilatedResidualConvBlock<T>? _decoderPostBlock;
    private DenseLayer<T>? _melProjection;

    // Duration predictor: residual blocks with kernels 4, 3, 1 -> FC to one log duration per phoneme.
    private readonly List<DilatedResidualConvBlock<T>> _durationBlocks = new();
    private DenseLayer<T>? _durationProjection;

    public override ModelOptions GetOptions() => _options;

    public SpeedySpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeedySpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeedySpeechOptions();
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

    public SpeedySpeech(
        NeuralNetworkArchitecture<T> architecture,
        SpeedySpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SpeedySpeechOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        // Adam with default parameters at a base rate of 0.002, gradients clipped to norm 1 (§4.3).
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, UseAdaptiveBetas = false }));
        MaxGradNorm = NumOps.FromDouble(_options.GradientClipNorm);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a log-mel spectrogram from text (pair it with a vocoder; the paper uses MelGAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>Durations come from the teacher network's attention (§3.1), outside the student.</remarks>
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

        int c = _options.HiddenDim;
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _embedding = new EmbeddingLayer<T>(_options.VocabSize, c);
        _prenet = new DenseLayer<T>(c, new ReLUActivation<T>() as IActivationFunction<T>);
        foreach (int d in _options.EncoderDilations)
            _encoderBlocks.Add(new DilatedResidualConvBlock<T>(c, _options.EncoderKernelSize, d, 2));
        _encoderPost1 = new DenseLayer<T>(c, identity);
        _encoderPostNorm = new BatchNormalizationLayer<T>(c, epsilon: 1e-5, momentum: 0.9);
        _encoderPost2 = new DenseLayer<T>(c, identity);

        foreach (int d in _options.DecoderDilations)
            _decoderBlocks.Add(new DilatedResidualConvBlock<T>(c, _options.DecoderKernelSize, d, 2));
        _decoderPost1 = new DenseLayer<T>(c, identity);
        _decoderPostBlock = new DilatedResidualConvBlock<T>(c, _options.DecoderKernelSize, 1, 2);
        _melProjection = new DenseLayer<T>(_options.MelChannels, identity);

        foreach (int k in _options.DurationPredictorKernelSizes)
            _durationBlocks.Add(new DilatedResidualConvBlock<T>(c, k, 1, 1));
        _durationProjection = new DenseLayer<T>(1, identity);

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_encoderPost1);
        encoder.Add(_encoderPostNorm);
        encoder.Add(_encoderPost2);
        var decoder = new List<ILayer<T>>(_decoderBlocks) { _decoderPost1, _decoderPostBlock, _melProjection };
        AddEncoderDecoderLayers(encoder, decoder);
        ComponentLayers.AddRange(_durationBlocks);
        ComponentLayers.Add(_durationProjection);
    }

    private bool HasPaperLayers => _embedding is not null;

    /// <summary>The phoneme encoder (Fig. 3): <c>[tokens] → [tokens, channels]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> tokens)
    {
        var prenet = _prenet!.Forward(_embedding!.Forward(tokens));
        var x = prenet;
        foreach (var block in _encoderBlocks) x = block.Forward(x);
        x = Engine.TensorAdd(_encoderPost1!.Forward(x), prenet);
        x = _encoderPostNorm!.Forward(Engine.ReLU(x));
        return _encoderPost2!.Forward(x);
    }

    /// <summary>Log durations per phoneme, from encodings whose gradient is detached (§3.2).</summary>
    private Tensor<T> PredictLogDurations(Tensor<T> encodings)
    {
        var x = new Tensor<T>(encodings._shape, encodings.ToVector());
        foreach (var block in _durationBlocks) x = block.Forward(x);
        return Engine.Reshape(_durationProjection!.Forward(x), new[] { encodings.Shape[0] });
    }

    /// <summary>The decoder (Fig. 3) on expanded encodings with phoneme-local positions: <c>[frames, channels] → [frames, mel]</c>.</summary>
    private Tensor<T> Decode(Tensor<T> expanded)
    {
        var x = expanded;
        foreach (var block in _decoderBlocks) x = block.Forward(x);
        x = Engine.TensorAdd(_decoderPost1!.Forward(x), expanded);
        return _melProjection!.Forward(_decoderPostBlock!.Forward(x));
    }

    /// <summary>
    /// Copies each phoneme encoding <c>d</c> times and adds a sinusoidal positional encoding that restarts at every
    /// phoneme (§3.2: "we reset the encoding for each phoneme"), interleaving sine and cosine channels as the authors'
    /// implementation does.
    /// </summary>
    private Tensor<T> Expand(Tensor<T> encodings, int[] durations)
    {
        var expanded = LengthRegulator.Expand(encodings, durations);
        int frames = expanded.Shape[0], channels = expanded.Shape[1];
        var positions = new Tensor<T>(new[] { frames, channels });
        int frame = 0;
        foreach (int d in durations)
            for (int p = 0; p < d; p++, frame++)
                for (int i = 0; i < channels / 2; i++)
                {
                    double angle = p / Math.Pow(10000.0, 2.0 * i / channels);
                    positions[frame, 2 * i] = NumOps.FromDouble(Math.Sin(angle));
                    positions[frame, 2 * i + 1] = NumOps.FromDouble(Math.Cos(angle));
                }
        return Engine.TensorAdd(expanded, positions);
    }

    /// <summary>Inference durations: <c>exp</c> of the prediction, at least one frame, rounded.</summary>
    private int[] InferenceDurations(Tensor<T> logDurations)
    {
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)Math.Round(Math.Max(1.0, Math.Exp(NumOps.ToDouble(logDurations[i]))));
        return durations;
    }

    /// <inheritdoc />
    /// <remarks>Returns the log-mel spectrogram, de-standardized with the dataset statistics the targets were
    /// standardized with.</remarks>
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
            throw new ArgumentException($"Expected tokens [tokens] or [batch, tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));

        // Each utterance runs alone (batch normalization uses running statistics at inference, so this matches a padded
        // batch); outputs are zero-padded to the longest.
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
        var encodings = Encode(tokens);
        var durations = InferenceDurations(PredictLogDurations(encodings));
        var standardized = Decode(Expand(encodings, durations));
        return Engine.TensorAddScalar(Engine.TensorMultiplyScalar(standardized, NumOps.FromDouble(_options.MelStd)),
            NumOps.FromDouble(_options.MelMean));
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
        var (target, objective) = BuildObjective(sample);
        return TrainWithCustomObjective(sample.Tokens, target, objective, _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (target, objective) = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, target)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The student's objective (§3.2): MAE plus <c>1 − SSIM</c> on the standardized log-mel spectrogram, and the Huber
    /// loss of the predicted log durations against <c>log(max(d, 1))</c>, with ground-truth durations driving the
    /// expansion.
    /// </summary>
    private (Tensor<T> Target, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(SpeedySpeech<T>)} trains on teacher-extracted durations; set {nameof(sample.Durations)}.", nameof(sample));
        var mel = DeriveAcousticTargets(sample).Mel;
        if (durations.Sum() != mel.Shape[0])
            throw new ArgumentException(
                $"The durations sum to {durations.Sum()} frames but the mel spectrogram has {mel.Shape[0]}.", nameof(sample));

        var standardized = new Tensor<T>(mel._shape);
        for (int i = 0; i < mel.Length; i++)
            standardized[i] = NumOps.FromDouble((NumOps.ToDouble(mel[i]) - _options.MelMean) / _options.MelStd);
        var logDurationTarget = new Tensor<T>(new[] { durations.Length });
        for (int i = 0; i < durations.Length; i++) logDurationTarget[i] = NumOps.FromDouble(Math.Log(Math.Max(1, durations[i])));

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            var encodings = Encode(tokens);
            var predicted = Decode(Expand(encodings, durations));
            var l1 = Mean(Engine.TensorAbs(Engine.TensorSubtract(predicted, target)));
            var ssim = SpectrogramSsim.Ssim(Engine, predicted, target);
            var dissimilarity = Engine.TensorAddScalar(Engine.TensorNegate(ssim), NumOps.One);
            var huber = Mean(SmoothL1(Engine.TensorSubtract(PredictLogDurations(encodings), logDurationTarget)));
            return Engine.TensorAdd(Engine.TensorAdd(l1, dissimilarity), huber);
        }

        return (standardized, Objective);
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>PyTorch's smooth L1 (Huber with δ = 1): <c>0.5 x²</c> where <c>|x| &lt; 1</c>, else <c>|x| − 0.5</c>.</summary>
    private Tensor<T> SmoothL1(Tensor<T> diff)
    {
        var abs = Engine.TensorAbs(diff);
        var quadratic = Engine.TensorMultiplyScalar(Engine.TensorMultiply(diff, diff), NumOps.FromDouble(0.5));
        var linear = Engine.TensorAddScalar(abs, NumOps.FromDouble(-0.5));
        var mask = new Tensor<T>(diff._shape);
        for (int i = 0; i < diff.Length; i++)
            mask[i] = Math.Abs(NumOps.ToDouble(diff[i])) < 1.0 ? NumOps.One : NumOps.Zero;
        var inverse = Engine.TensorAddScalar(Engine.TensorNegate(mask), NumOps.One);
        return Engine.TensorAdd(Engine.TensorMultiply(mask, quadratic), Engine.TensorMultiply(inverse, linear));
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
            Name = _useNativeMode ? "SpeedySpeech-Native" : "SpeedySpeech-ONNX",
            Description = "SpeedySpeech: Efficient Neural Speech Synthesis (Vainer & Dusek, 2020)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.EncoderDilations.Length + _options.DecoderDilations.Length,
        };
        m.AdditionalInfo["Architecture"] = "SpeedySpeech";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeedySpeech<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
