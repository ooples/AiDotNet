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
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Glow-TTS: a flow-based generative text-to-speech model that finds its own alignment with monotonic alignment search.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"
/// (Kim et al., NeurIPS 2020) and its reference implementation (jaywalnut310/glow-tts) for what the paper leaves unstated.</para>
/// <para>
/// The text encoder (§3.3, Fig. 7) embeds phonemes, refines them with a residual convolutional pre-net and a Transformer
/// encoder with relative position representations (<see cref="RelativePositionTransformerBlock{T}"/>), and projects
/// each token to the mean of a Gaussian prior over mel frames (σ fixed at 1). The decoder (Fig. 8) is a normalizing
/// flow over pairs of frames squeezed into channels: blocks of <see cref="ActNormFlowLayer{T}"/>,
/// <see cref="GroupedInvertibleConvFlowLayer{T}"/> and <see cref="AffineCouplingFlowLayer{T}"/>. Training maximizes
/// the exact log-likelihood of the mel spectrogram under the prior aligned by monotonic alignment search (§3.2), plus a
/// duration predictor's log-domain MSE to the searched durations (on encoder outputs with their gradient stopped).
/// Inference samples the expanded prior at temperature 0.333 and runs the flow in reverse.
/// </para>
/// <para>The model aligns itself, so a plain <c>Train(tokens, mel)</c> is its full training signal.</para>
/// <para><b>For Beginners:</b> Glow-TTS learns an exactly reversible mapping between spectrograms and simple Gaussian
/// noise shaped by the text; to speak, it draws noise shaped by the text and runs the mapping backwards.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search",
    "https://arxiv.org/abs/2005.11129",
    Year = 2020,
    Authors = "Kim et al."
)]
[PaperOptimizer(OptimizerKind.Adam,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000,
                Source = "Kim et al. 2020, Sec. 4: Adam with the Noam learning rate schedule, trained for 240K iterations. No peak rate is declared because the paper states none, and the Noam peak is a function of the model dimension; the warmup is the schedule default of 4000 from Vaswani et al. 2017, which the paper cites without restating a length.")]
public partial class GlowTTS<T> : TtsModelBase<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly GlowTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Encoder (§3.3, Fig. 7): embedding -> pre-net -> relative-position Transformer -> prior means; duration predictor.
    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _prenet;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _meanProjection;
    private VariancePredictorLayer<T>? _durationPredictor;

    // Decoder (Fig. 8): squeeze, n x (ActNorm, grouped invertible 1x1 conv, affine coupling), unsqueeze.
    private readonly List<LayerBase<T>> _flows = new();

    public override ModelOptions GetOptions() => _options;

    public GlowTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, GlowTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GlowTTSOptions();
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

    public GlowTTS(
        NeuralNetworkArchitecture<T> architecture,
        GlowTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GlowTTSOptions();
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

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with WaveGlow).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

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
        int h = o.HiddenDim;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, h);
        _prenet = new ResidualConvReluNormLayer<T>(h, o.PrenetKernelSize, o.PrenetLayers, o.PrenetDropout);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new RelativePositionTransformerBlock<T>(h, o.NumHeads, o.FilterChannels, o.EncoderKernelSize,
                o.DropoutRate, o.RelativeWindow));
        _meanProjection = new DenseLayer<T>(o.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _durationPredictor = new VariancePredictorLayer<T>(h, o.DurationPredictorFilterChannels, 1, o.EncoderKernelSize, o.DropoutRate);

        int flowChannels = o.MelChannels * o.SqueezeFactor;
        for (int b = 0; b < o.NumFlowBlocks; b++)
        {
            _flows.Add(new ActNormFlowLayer<T>(flowChannels));
            _flows.Add(new GroupedInvertibleConvFlowLayer<T>(flowChannels, o.InvertibleConvGroupSize));
            _flows.Add(new AffineCouplingFlowLayer<T>(flowChannels, o.DecoderHiddenChannels, o.DecoderKernelSize,
                o.DecoderDilationRate, o.CouplingLayers, o.DecoderDropout));
        }

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_meanProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(_flows);
    }

    private bool HasPaperLayers => _embedding is not null;

    /// <summary>
    /// The text encoder: returns the encoder hidden sequence <c>[tokens, hidden]</c> and the prior means
    /// <c>[tokens, melChannels]</c>. Embeddings are scaled by √hidden as in the reference implementation.
    /// </summary>
    private (Tensor<T> Hidden, Tensor<T> Means) Encode(Tensor<T> tokens)
    {
        var x = Engine.TensorMultiplyScalar(_embedding!.Forward(tokens), NumOps.FromDouble(Math.Sqrt(_options.HiddenDim)));
        x = _prenet!.Forward(x);
        foreach (var block in _encoderBlocks) x = block.Forward(x);
        return (x, _meanProjection!.Forward(x));
    }

    /// <summary>Predicted log durations from the encoder output, with its gradient stopped (§3.1).</summary>
    private Tensor<T> LogDurations(Tensor<T> hidden)
    {
        var detached = new Tensor<T>(hidden._shape, hidden.ToVector());
        return Engine.Reshape(_durationPredictor!.Forward(detached), new[] { hidden.Shape[0] });
    }

    /// <summary>
    /// The flow decoder on <c>[frames, mel]</c> (frames even): squeeze pairs of frames into channels, run the flows
    /// forward (data → latent, returning the summed log-determinant) or in reverse, and unsqueeze.
    /// </summary>
    private (Tensor<T> Output, Tensor<T>? LogDeterminant) Flow(Tensor<T> frames, bool reverse)
    {
        int length = frames.Shape[0], channels = frames.Shape[1], sqz = _options.SqueezeFactor;
        // [frames, mel] -> [1, mel, frames] -> [1, mel * sqz, frames / sqz] (channel index = sub-frame * mel + bin).
        var x = Engine.Reshape(Engine.TensorTranspose(frames), new[] { 1, channels, length / sqz, sqz });
        x = Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 3, 1, 2 }).Contiguous(), new[] { 1, channels * sqz, length / sqz });

        Tensor<T>? logDet = null;
        var steps = reverse ? Enumerable.Reverse(_flows) : _flows;
        foreach (var step in steps)
        {
            var (y, stepLogDet) = ((IInvertibleFlowStep<T>)step).Transform(x, reverse);
            x = y;
            if (stepLogDet is not null) logDet = logDet is null ? stepLogDet : Engine.TensorAdd(logDet, stepLogDet);
        }

        // Unsqueeze: [1, sqz * mel, frames / sqz] -> [1, sqz, mel, frames / sqz] -> [1, mel, frames / sqz, sqz] -> [frames, mel].
        var unsqueezed = Engine.TensorPermute(Engine.Reshape(x, new[] { 1, sqz, channels, length / sqz }), new[] { 0, 2, 3, 1 }).Contiguous();
        var output = Engine.TensorTranspose(Engine.Reshape(unsqueezed, new[] { channels, length }));
        return (output, logDet);
    }

    /// <inheritdoc />
    /// <remarks>Inference (§3.1): durations ⌈exp(log d̂) · length scale⌉, the prior means expanded by them, a sample
    /// <c>z = μ + T · ε</c> at temperature T (0.333 by default), and the decoder run in reverse.</remarks>
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
        var (hidden, means) = Encode(tokens);
        var logDurations = LogDurations(hidden);
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)Math.Ceiling(Math.Exp(NumOps.ToDouble(logDurations[i])) * _options.LengthScale);
        int sqz = _options.SqueezeFactor;
        int total = durations.Sum();
        // The decoder squeezes frame pairs: the reference implementation floors the length to a multiple of the factor.
        int frames = Math.Max(sqz, total / sqz * sqz);
        if (total < frames) durations[durations.Length - 1] += frames - total;
        var prior = LengthRegulator.Expand(means, durations);
        prior = Engine.TensorSlice(prior, new[] { 0, 0 }, new[] { frames, _options.MelChannels });

        // z = mu + T * eps, with a fixed sampling seed so that synthesis is repeatable.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var noise = new Tensor<T>(prior._shape);
        for (int i = 0; i < noise.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            noise[i] = NumOps.FromDouble(_options.Temperature * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return Flow(Engine.TensorAdd(prior, noise), reverse: true).Output;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }
        var (tokens, mel) = Prepare(input, expectedOutput);
        TrainWithCustomObjective(tokens, mel, Objective, _optimizer);
    }

    /// <inheritdoc />
    /// <remarks>Glow-TTS finds its own alignment (monotonic alignment search), so a token/mel pair is its whole supervision.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        Train(sample.Tokens, DeriveAcousticTargets(sample).Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
        => ((ITrainingObjectiveProvider<T>)this).EvaluateTrainingObjective(sample.Tokens, DeriveAcousticTargets(sample).Mel);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var (tokens, mel) = Prepare(input, target);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private (Tensor<T> Tokens, Tensor<T> Mel) Prepare(Tensor<T> input, Tensor<T> target)
    {
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        var mel = target.Rank == 3 ? Engine.Reshape(target, new[] { target.Shape[1], target.Shape[2] }) : target;
        if (tokens.Rank != 1 || mel.Rank != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException(
                $"Expected tokens [tokens] and a mel spectrogram [frames, {_options.MelChannels}], got [{string.Join(", ", input.Shape)}] and [{string.Join(", ", target.Shape)}].");
        int sqz = _options.SqueezeFactor, frames = mel.Shape[0] / sqz * sqz;
        if (frames < tokens.Shape[0])
            throw new ArgumentException(
                $"Monotonic alignment gives every token at least one frame: {tokens.Shape[0]} tokens need at least that many frames, got {frames}.");
        if (frames != mel.Shape[0])
            mel = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
        return (tokens, mel);
    }

    /// <summary>
    /// The training objective (§3.1–3.2): the negative log-likelihood per mel element,
    /// <c>[½ Σ (z − μ_A)² − log|det ∂f⁻¹/∂x|] / (F · C) + ½ log 2π</c> with σ fixed at 1, where the alignment A is
    /// the monotonic alignment search over the prior log-likelihoods (no gradient); plus the duration predictor's
    /// mean squared error against <c>log(1e-8 + d_A)</c>.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var (hidden, means) = Encode(tokens);
        var (z, logDet) = Flow(mel, reverse: false);

        // log N(z_j; mu_i, 1) up to a constant, for the search only.
        int tokenCount = means.Shape[0];
        var scores = new double[tokenCount, frames];
        for (int i = 0; i < tokenCount; i++)
            for (int j = 0; j < frames; j++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    double d = NumOps.ToDouble(z[j, c]) - NumOps.ToDouble(means[i, c]);
                    sum += d * d;
                }
                scores[i, j] = -0.5 * sum;
            }
        int[] durations = MonotonicAlignment.MaximumPath(scores);

        var alignedMeans = LengthRegulator.Expand(means, durations);
        var diff = Engine.TensorSubtract(z, alignedMeans);
        var squared = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
        var negLogLikelihood = Engine.TensorSubtract(Engine.TensorMultiplyScalar(squared, NumOps.FromDouble(0.5)), logDet!);
        var mle = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(negLogLikelihood, NumOps.FromDouble(1.0 / (frames * channels))),
            NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI)));

        var target = new Tensor<T>(new[] { tokenCount });
        for (int i = 0; i < tokenCount; i++) target[i] = NumOps.FromDouble(Math.Log(1e-8 + durations[i]));
        var durationError = Engine.TensorSubtract(LogDurations(hidden), target);
        var durationLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false),
            NumOps.FromDouble(1.0 / tokenCount));
        return Engine.TensorAdd(mle, durationLoss);
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
            Name = _useNativeMode ? "GlowTTS-Native" : "GlowTTS-ONNX",
            Description = "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search (Kim et al., 2020)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumFlowBlocks,
        };
        m.AdditionalInfo["Architecture"] = "GlowTTS";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GlowTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
