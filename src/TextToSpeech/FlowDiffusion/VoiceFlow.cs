using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// VoiceFlow: a text-to-speech acoustic model that generates mel spectrograms with conditional flow matching and
/// straightens its sampling trajectory with rectified flow.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching" (Guo et al., ICASSP
/// 2024) and its reference implementation (cantabile-kwok/VoiceFlow-TTS) for what the paper leaves unstated.</para>
/// <para>
/// The text encoder and duration predictor are Grad-TTS's; durations come from a forced alignment (§3.1, §4.1) and
/// duplicate the encoder output into the frame-level condition y. A Grad-TTS U-Net u_θ(x_t, y, t) estimates the vector
/// field of the path <c>p_t(x | x₀, x₁) = N(t x₁ + (1 − t) x₀, σ² I)</c>, x₀ ~ N(0, I), whose target is the straight
/// line <c>x₁ − x₀</c> (Eq. 4–5); the loss is <c>L_FM + L_dur</c>. Sampling solves <c>dx = u_θ dt</c> with N Euler
/// steps from t = 0 to 1 (Eq. 6). Flow rectification (§3.2, Algorithm 1) samples a noise x₀′ per utterance, solves the
/// ODE with the ground-truth durations to obtain x̂₁ (<see cref="GenerateRectificationPair"/>), and trains again on the
/// fixed pair (<see cref="TrainRectified"/>, Eq. 7).
/// </para>
/// <para><b>For Beginners:</b> VoiceFlow learns to move random noise in a straight line to a spectrogram; training a
/// second time on its own noise/output pairs makes the lines straighter, so very few steps are needed.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching",
    "https://arxiv.org/abs/2309.05027",
    Year = 2024,
    Authors = "Guo et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, ReferenceBatchSize = 24,
                Provenance = RecipeProvenance.DerivedFromCitedWork,
                Source = "The paper does not state its optimizer; the reference implementation trains with "
                        + "torch.optim.Adam at 5e-5 and a batch of 24 (configs/lj_16k_gt_dur.yaml), "
                        + "following the Grad-TTS recipe it builds on.")]
public partial class VoiceFlow<T> : TtsModelBase<T>, IEndToEndTts<T>, IAcousticModel<T>
{
    private readonly VoiceFlowOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;

    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _prenet;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _meanProjection;
    private VariancePredictorLayer<T>? _durationPredictor;
    private GradTtsScoreEstimator<T>? _estimator;

    public override ModelOptions GetOptions() => _options;

    public VoiceFlow(NeuralNetworkArchitecture<T> architecture, string modelPath, VoiceFlowOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VoiceFlowOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
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

    public VoiceFlow(
        NeuralNetworkArchitecture<T> architecture,
        VoiceFlowOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VoiceFlowOptions();
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
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
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumFlowSteps;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with HiFi-GAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>Durations come from a forced alignment (§3.1: "an explicit duration learning module from forced
    /// alignments").</remarks>
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
        int h = o.HiddenDim;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, h);
        _prenet = new ResidualConvReluNormLayer<T>(h, o.PrenetKernelSize, o.PrenetLayers, o.PrenetDropout);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new RelativePositionTransformerBlock<T>(h, o.NumHeads, o.FilterChannels, o.EncoderKernelSize,
                o.DropoutRate, o.RelativeWindow));
        _meanProjection = new DenseLayer<T>(o.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _durationPredictor = new VariancePredictorLayer<T>(h, o.DurationPredictorFilterChannels, 1, o.EncoderKernelSize, o.DropoutRate);
        _estimator = new GradTtsScoreEstimator<T>(Engine, o.FlowDim, o.DecoderDimMultipliers, o.TimePositionScale);

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_meanProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(_estimator.Layers);
    }

    private bool HasPaperLayers => _embedding is not null;

    private (Tensor<T> Hidden, Tensor<T> Means) Encode(Tensor<T> tokens)
    {
        var x = Engine.TensorMultiplyScalar(_embedding!.Forward(tokens), NumOps.FromDouble(Math.Sqrt(_options.HiddenDim)));
        x = _prenet!.Forward(x);
        foreach (var block in _encoderBlocks) x = block.Forward(x);
        return (x, _meanProjection!.Forward(x));
    }

    private Tensor<T> LogDurations(Tensor<T> hidden)
    {
        var detached = new Tensor<T>(hidden._shape, hidden.ToVector());
        return Engine.Reshape(_durationPredictor!.Forward(detached), new[] { hidden.Shape[0] });
    }

    /// <inheritdoc />
    /// <remarks>Encoder and decoder gradients are each clipped to norm 1 (reference train.py).</remarks>
    protected override IReadOnlyList<IReadOnlyList<Tensor<T>>>? GradientClippingGroups(IReadOnlyList<Tensor<T>> trainableParameters)
    {
        if (!HasPaperLayers) return null;
        var encoderLayers = Layers.Take(EncoderLayerCount).Append(_durationPredictor!).Cast<ILayer<T>>().ToList();
        var encoder = Training.TapeTrainingStep<T>.CollectParameters(encoderLayers, -1);
        var decoder = Training.TapeTrainingStep<T>.CollectParameters(_estimator!.Layers.Cast<ILayer<T>>().ToList(), -1);
        return new IReadOnlyList<Tensor<T>>[] { encoder, decoder };
    }

    private static int CompatibleLength(int length) => (length + 3) / 4 * 4;

    private Tensor<T> FrameMask(int frames, int padded)
    {
        var mask = new Tensor<T>(new[] { 1, 1, 1, padded });
        for (int i = 0; i < frames; i++) mask[0, 0, 0, i] = NumOps.One;
        return mask;
    }

    private Tensor<T> MaskRows(Tensor<T> mask, int padded)
        => Engine.TensorTranspose(Engine.TensorTile(Engine.Reshape(mask, new[] { 1, padded }), new[] { _options.MelChannels, 1 }));

    private Tensor<T> PadFrames(Tensor<T> x, int frames)
    {
        if (x.Shape[0] == frames) return x;
        return Engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { frames - x.Shape[0], x.Shape[1] }) }, 0);
    }

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    /// <summary>N Euler steps of dx = u_θ(x, y, t) dt from <paramref name="x0"/> at t = 0 (Eq. 6).</summary>
    private Tensor<T> Solve(Tensor<T> x0, Tensor<T> condition, Tensor<T> mask, Tensor<T> maskRows, int steps)
    {
        var x = Engine.TensorMultiply(x0, maskRows);
        for (int k = 0; k < steps; k++)
        {
            var velocity = _estimator!.Estimate(x, condition, (double)k / steps, mask);
            x = Engine.TensorMultiply(Engine.TensorAdd(x, Engine.TensorMultiplyScalar(velocity, NumOps.FromDouble(1.0 / steps))), maskRows);
        }
        return x;
    }

    // ---------------------------------------------------------------- inference

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
        if (input.Rank == 2 && input.Shape[0] == 1)
            input = Engine.Reshape(input, new[] { input.Shape[1] });
        if (input.Rank != 1)
            throw new ArgumentException($"Expected tokens [tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));

        using var _ = new NoGradScope<T>();
        var (hidden, means) = Encode(input);
        var logDurations = LogDurations(hidden);
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)(Math.Ceiling(Math.Exp(NumOps.ToDouble(logDurations[i]))) * _options.LengthScale);
        if (durations.Sum() == 0) durations[durations.Length - 1] = 1;
        int frames = durations.Sum(), padded = CompatibleLength(frames);
        var condition = PadFrames(LengthRegulator.Expand(means, durations), padded);
        var mask = FrameMask(frames, padded);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var mel = Solve(Gaussian(condition._shape, random), condition, mask, MaskRows(mask, padded), Math.Max(1, _options.NumFlowSteps));
        return Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
    }

    // ---------------------------------------------------------------- training

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
        var (mel, objective) = BuildObjective(sample, null, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, mel, objective, _optimizer);
    }

    /// <inheritdoc />
    /// <remarks>t, the noise and the segment are fixed by the sampling seed so the same parameters always score the same.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (mel, objective) = BuildObjective(sample, null, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
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
    /// Flow rectification, data generation (Algorithm 1 lines 10–11): draws a noise x₀′ for the utterance and solves the
    /// ODE from it with the ground-truth durations, returning the pair (x₀′, x̂₁), both <c>[frames, mel]</c>.
    /// </summary>
    /// <param name="sample">The utterance; its durations drive the length regulation.</param>
    /// <param name="steps">Euler steps for the generation (the trained model's N by default).</param>
    public (Tensor<T> Noise, Tensor<T> Mel) GenerateRectificationPair(TtsTrainingSample<T> sample, int? steps = null)
    {
        ThrowIfDisposed();
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("Rectification needs the paper's layers; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(VoiceFlow<T>)} generates rectification data with ground-truth durations; set {nameof(sample.Durations)}.", nameof(sample));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var (_, means) = Encode(sample.Tokens);
            int frames = durations.Sum(), padded = CompatibleLength(frames);
            var condition = PadFrames(LengthRegulator.Expand(means, durations), padded);
            var mask = FrameMask(frames, padded);
            var noise = Engine.TensorMultiply(Gaussian(condition._shape, _trainingRandom), MaskRows(mask, padded));
            var mel = Solve(noise, condition, mask, MaskRows(mask, padded), Math.Max(1, steps ?? _options.NumFlowSteps));
            return (Engine.TensorSlice(noise, new[] { 0, 0 }, new[] { frames, _options.MelChannels }),
                Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { frames, _options.MelChannels }));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Flow rectification, training (Eq. 7): one step on the fixed pair — <paramref name="sample"/>'s mel is the
    /// generated x̂₁ and <paramref name="noise"/> the x₀′ it was generated from, both <c>[frames, mel]</c>.
    /// </summary>
    /// <returns>The training loss of the step.</returns>
    public T TrainRectified(TtsTrainingSample<T> sample, Tensor<T> noise)
    {
        ThrowIfDisposed();
        Guard.NotNull(noise);
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var (mel, objective) = BuildObjective(sample, noise, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, mel, objective, _optimizer);
    }

    /// <summary>
    /// The objective (§3.1, Algorithm 1 TrainStep): <c>L_dur</c>, the mean squared log-duration error against the
    /// forced-alignment durations, plus <c>L_FM = mean (u_θ(x_t, y, t) − (x₁ − x₀))²</c> on a 2-second segment with
    /// x_t = t x₁ + (1 − t) x₀ + σ ε and t ~ U[1e-5, 1 − 1e-5]; x₀ is fresh N(0, I) noise, or the paired noise when
    /// rectifying.
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample,
        Tensor<T>? pairedNoise, Random random)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(VoiceFlow<T>)} trains on forced-alignment durations; set {nameof(sample.Durations)}.", nameof(sample));
        var targets = DeriveAcousticTargets(sample);
        int frames = targets.MelFrames, channels = _options.MelChannels;
        if (durations.Sum() != frames)
            throw new ArgumentException($"The durations sum to {durations.Sum()} frames but the mel spectrogram has {frames}.", nameof(sample));
        if (pairedNoise is not null && (pairedNoise.Shape[0] != frames || pairedNoise.Shape[1] != channels))
            throw new ArgumentException($"The paired noise must be [{frames}, {channels}].", nameof(pairedNoise));

        double t = Math.Min(Math.Max(random.NextDouble(), 1e-5), 1.0 - 1e-5);
        int length = Math.Min(frames, _options.SegmentFrames);
        int offset = frames > _options.SegmentFrames ? random.Next(0, frames - _options.SegmentFrames) : 0;
        int padded = CompatibleLength(length);
        var freshNoise = Gaussian(new[] { padded, channels }, random);
        var pathNoise = Gaussian(new[] { padded, channels }, random);
        var logTargets = new Tensor<T>(new[] { durations.Length });
        for (int i = 0; i < durations.Length; i++) logTargets[i] = NumOps.FromDouble(Math.Log(1e-8 + durations[i]));

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel)
        {
            var (hidden, means) = Encode(tokens);
            var durationError = Engine.TensorSubtract(LogDurations(hidden), logTargets);
            var durationLoss = Engine.ReduceMean(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false);

            var mask = FrameMask(length, padded);
            var maskRows = MaskRows(mask, padded);
            var x1 = PadFrames(Engine.TensorSlice(mel, new[] { offset, 0 }, new[] { length, channels }), padded);
            var condition = PadFrames(Engine.TensorSlice(LengthRegulator.Expand(means, durations), new[] { offset, 0 }, new[] { length, channels }), padded);
            var x0 = Engine.TensorMultiply(pairedNoise is null
                ? freshNoise
                : PadFrames(Engine.TensorSlice(pairedNoise, new[] { offset, 0 }, new[] { length, channels }), padded), maskRows);
            var xt = Engine.TensorAdd(Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(x1, NumOps.FromDouble(t)),
                    Engine.TensorMultiplyScalar(x0, NumOps.FromDouble(1 - t))),
                Engine.TensorMultiplyScalar(pathNoise, NumOps.FromDouble(_options.Sigma)));
            var residual = Engine.TensorSubtract(_estimator!.Estimate(xt, condition, t, mask), Engine.TensorSubtract(x1, x0));
            var flowLoss = Engine.ReduceMean(Engine.TensorMultiply(residual, residual), new[] { 0, 1 }, keepDims: false);
            return Engine.TensorAdd(flowLoss, durationLoss);
        }

        return (targets.Mel, Objective);
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
            Name = _useNativeMode ? "VoiceFlow-Native" : "VoiceFlow-ONNX",
            Description = "VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching (Guo et al., 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "VoiceFlow";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VoiceFlow<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
