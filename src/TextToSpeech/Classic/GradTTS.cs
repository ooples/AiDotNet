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
/// Grad-TTS: text-to-speech with a score-based diffusion decoder that turns an aligned Gaussian prior into a mel
/// spectrogram.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech" (Popov et al., ICML 2021)
/// and its reference implementation (huawei-noah/Speech-Backbones) for what the paper leaves unstated.</para>
/// <para>
/// The encoder is Glow-TTS's (§3): it predicts per-token prior means μ and durations, aligned to the mel spectrogram
/// by monotonic alignment search during training. The decoder is a score network s_θ(X_t, μ, t)
/// (<c>GradTtsScoreEstimator</c>, a U-Net) for the forward diffusion
/// <c>dX = ½ β_t (μ − X) dt + √β_t dW</c> with β_t = 0.05 + 19.95 t, whose terminal distribution is N(μ, I).
/// Training minimizes the duration loss, the prior loss ½‖y − μ‖² and the score-matching loss on random 2-second
/// segments (MAS and durations use the whole utterance), with encoder and decoder gradients each clipped to norm 1.
/// Inference draws X₁ = μ + ε/τ (τ = 1.5) and solves the reverse ODE with N Euler steps.
/// </para>
/// <para>The model aligns itself, so a plain <c>Train(tokens, mel)</c> is its full training signal.</para>
/// <para><b>For Beginners:</b> Grad-TTS first sketches a blurry spectrogram from the text, then refines it step by step
/// by removing noise, the way image diffusion models paint pictures.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech",
    "https://arxiv.org/abs/2105.06337",
    Year = 2021,
    Authors = "Popov et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.0001,
                Source = "Popov et al. 2021, Sec. 4: Adam with the learning rate set to 0.0001.")]
public partial class GradTTS<T> : TtsModelBase<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly GradTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;

    // Encoder: Glow-TTS's text encoder (§3, "encoder architecture from Glow-TTS"): embedding, conv pre-net,
    // relative-position Transformer, prior means, duration predictor on stop-gradient encoder output.
    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _prenet;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _meanProjection;
    private VariancePredictorLayer<T>? _durationPredictor;

    // Decoder: the U-Net score network s_theta(X_t, mu, t).
    private GradTtsScoreEstimator<T>? _scoreNetwork;

    public override ModelOptions GetOptions() => _options;

    public GradTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, GradTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GradTTSOptions();
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

    public GradTTS(
        NeuralNetworkArchitecture<T> architecture,
        GradTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GradTTSOptions();
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
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with HiFi-GAN).</summary>
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
        _scoreNetwork = new GradTtsScoreEstimator<T>(Engine, o.DecoderDim, o.DecoderDimMultipliers, o.TimePositionScale);

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_meanProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(_scoreNetwork.Layers);
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
    /// <remarks>Each of the encoder and the decoder is clipped to norm 1 on its own, as the reference training loop
    /// does (<c>clip_grad_norm_(model.encoder.parameters(), 1)</c>, then the decoder's).</remarks>
    protected override IReadOnlyList<IReadOnlyList<Tensor<T>>>? GradientClippingGroups(IReadOnlyList<Tensor<T>> trainableParameters)
    {
        if (!HasPaperLayers) return null;
        var encoderLayers = Layers.Take(EncoderLayerCount).Append(_durationPredictor!).Cast<ILayer<T>>().ToList();
        var encoder = Training.TapeTrainingStep<T>.CollectParameters(encoderLayers, -1);
        var decoder = Training.TapeTrainingStep<T>.CollectParameters(_scoreNetwork!.Layers.Cast<ILayer<T>>().ToList(), -1);
        return new IReadOnlyList<Tensor<T>>[] { encoder, decoder };
    }

    // cumulative noise ∫_0^t β_s ds and β_t for β_t = β0 + (β1 − β0) t.
    private double CumulativeNoise(double t) => _options.BetaMin * t + 0.5 * (_options.BetaMax - _options.BetaMin) * t * t;
    private double Noise(double t) => _options.BetaMin + (_options.BetaMax - _options.BetaMin) * t;

    /// <summary>Smallest multiple of 4 (two U-Net downsamplings) not below <paramref name="length"/>.</summary>
    private static int CompatibleLength(int length) => (length + 3) / 4 * 4;

    private Tensor<T> FrameMask(int frames, int padded)
    {
        var mask = new Tensor<T>(new[] { 1, 1, 1, padded });
        for (int i = 0; i < frames; i++) mask[0, 0, 0, i] = NumOps.One;
        return mask;
    }

    private Tensor<T> PadFrames(Tensor<T> x, int frames)
    {
        if (x.Shape[0] == frames) return x;
        return Engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { frames - x.Shape[0], x.Shape[1] }) }, 0);
    }

    private Tensor<T> Gaussian(int[] shape, Random random, double scale)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    /// <inheritdoc />
    /// <remarks>Inference (§3–4): durations ⌈exp(log d̂)⌉ · length scale, the prior expanded and padded to a multiple of
    /// 4 frames, the terminal sample <c>X_1 = μ + ε / τ</c> (τ = 1.5), and N steps (10) of the reverse ODE
    /// <c>dX = ½(μ − X − s_θ(X, μ, t)) β_t dt</c> solved backwards with Euler steps at the interval midpoints.</remarks>
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
        using var _ = new NoGradScope<T>();
        var (hidden, means) = Encode(tokens);
        var logDurations = LogDurations(hidden);
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)(Math.Ceiling(Math.Exp(NumOps.ToDouble(logDurations[i]))) * _options.LengthScale);
        int frames = Math.Max(1, durations.Sum());
        if (durations.Sum() == 0) durations[durations.Length - 1] = 1;
        int padded = CompatibleLength(frames);
        var mu = PadFrames(LengthRegulator.Expand(means, durations), padded);
        var mask = FrameMask(frames, padded);
        var maskRows = Engine.TensorTranspose(Engine.TensorTile(Engine.Reshape(mask, new[] { 1, padded }), new[] { _options.MelChannels, 1 }));

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var x = Engine.TensorMultiply(Engine.TensorAdd(mu, Gaussian(mu._shape, random, 1.0 / _options.Temperature)), maskRows);
        int steps = _options.NumDiffusionSteps;
        double h = 1.0 / steps;
        for (int i = 0; i < steps; i++)
        {
            double t = 1.0 - (i + 0.5) * h;
            var score = _scoreNetwork!.Estimate(x, mu, t, mask);
            var dx = Engine.TensorMultiplyScalar(Engine.TensorSubtract(Engine.TensorSubtract(mu, x), score), NumOps.FromDouble(0.5 * Noise(t) * h));
            x = Engine.TensorMultiply(Engine.TensorSubtract(x, dx), maskRows);
        }
        return Engine.TensorSlice(x, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
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
        var draw = DrawTrainingRandomness(mel.Shape[0], _trainingRandom);
        TrainWithCustomObjective(tokens, mel, (x, y) => Objective(x, y, draw), _optimizer);
    }

    /// <inheritdoc />
    /// <remarks>Grad-TTS finds its own alignment with monotonic alignment search, so a token/mel pair is its whole supervision.</remarks>
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

    /// <remarks>The diffusion loss is an expectation over t, the noise and the segment; evaluation fixes them with the
    /// sampling seed so the same parameters always score the same.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var (tokens, mel) = Prepare(input, target);
        var draw = DrawTrainingRandomness(mel.Shape[0], AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(tokens, mel, draw)[0];
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
        if (mel.Shape[0] < tokens.Shape[0])
            throw new ArgumentException(
                $"Monotonic alignment gives every token at least one frame: {tokens.Shape[0]} tokens need at least that many frames, got {mel.Shape[0]}.");
        return (tokens, mel);
    }

    private sealed record TrainingDraw(double Time, int SegmentOffset, Tensor<T> Noise);

    // t ~ U(1e-5, 1 - 1e-5), a random 2-second segment offset, and xi ~ N(0, I) over the segment.
    private TrainingDraw DrawTrainingRandomness(int frames, Random random)
    {
        double t = Math.Min(Math.Max(random.NextDouble(), 1e-5), 1.0 - 1e-5);
        int segment = Math.Min(frames, _options.SegmentFrames);
        int offset = frames > _options.SegmentFrames ? random.Next(0, frames - _options.SegmentFrames) : 0;
        var noise = Gaussian(new[] { CompatibleLength(Math.Max(segment, _options.SegmentFrames)), _options.MelChannels }, random, 1.0);
        return new TrainingDraw(t, offset, noise);
    }

    /// <summary>
    /// The training objective (§3, reference <c>compute_loss</c>): duration loss against the MAS durations over the whole
    /// mel, then on a fixed-length segment the prior loss <c>Σ ½((y − μ)² + log 2π) / (frames · mel)</c> and the diffusion
    /// loss <c>Σ (s_θ(X_t, μ, t) · √(1 − e^{−∫β}) + ξ)² / (frames · mel)</c> with
    /// <c>X_t = y e^{−½∫β} + μ (1 − e^{−½∫β}) + ξ √(1 − e^{−∫β})</c>.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel, TrainingDraw draw)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var (hidden, means) = Encode(tokens);

        int tokenCount = means.Shape[0];
        var scores = new double[tokenCount, frames];
        for (int i = 0; i < tokenCount; i++)
            for (int j = 0; j < frames; j++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    double d = NumOps.ToDouble(mel[j, c]) - NumOps.ToDouble(means[i, c]);
                    sum += d * d;
                }
                scores[i, j] = -0.5 * sum;
            }
        int[] durations = MonotonicAlignment.MaximumPath(scores);

        var target = new Tensor<T>(new[] { tokenCount });
        for (int i = 0; i < tokenCount; i++) target[i] = NumOps.FromDouble(Math.Log(1e-8 + durations[i]));
        var durationError = Engine.TensorSubtract(LogDurations(hidden), target);
        var durationLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false),
            NumOps.FromDouble(1.0 / tokenCount));

        // Segment of up to SegmentFrames frames, zero-padded to the segment length (rounded to a multiple of 4) and masked.
        int length = Math.Min(frames, _options.SegmentFrames);
        int padded = draw.Noise.Shape[0];
        var y = PadFrames(Engine.TensorSlice(mel, new[] { draw.SegmentOffset, 0 }, new[] { length, channels }), padded);
        var muFull = LengthRegulator.Expand(means, durations);
        var mu = PadFrames(Engine.TensorSlice(muFull, new[] { draw.SegmentOffset, 0 }, new[] { length, channels }), padded);
        var mask = FrameMask(length, padded);
        var maskRows = Engine.TensorTranspose(Engine.TensorTile(Engine.Reshape(mask, new[] { 1, padded }), new[] { channels, 1 }));
        double count = length * (double)channels;

        var priorDiff = Engine.TensorSubtract(y, mu);
        var priorSum = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(priorDiff, priorDiff), maskRows), new[] { 0, 1 }, keepDims: false);
        var priorLoss = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(priorSum, NumOps.FromDouble(0.5 / count)),
            NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI)));

        double cumulative = CumulativeNoise(draw.Time);
        double keep = Math.Exp(-0.5 * cumulative), std = Math.Sqrt(1.0 - Math.Exp(-cumulative));
        var noise = Engine.TensorMultiply(draw.Noise, maskRows);
        var xt = Engine.TensorMultiply(Engine.TensorAdd(Engine.TensorAdd(
            Engine.TensorMultiplyScalar(y, NumOps.FromDouble(keep)),
            Engine.TensorMultiplyScalar(mu, NumOps.FromDouble(1.0 - keep))),
            Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(std))), maskRows);
        var estimate = Engine.TensorMultiplyScalar(_scoreNetwork!.Estimate(xt, mu, draw.Time, mask), NumOps.FromDouble(std));
        var residual = Engine.TensorAdd(estimate, noise);
        var diffusionLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(residual, residual), new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(1.0 / count));

        return Engine.TensorAdd(Engine.TensorAdd(durationLoss, priorLoss), diffusionLoss);
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
            Name = _useNativeMode ? "GradTTS-Native" : "GradTTS-ONNX",
            Description = "Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech (Popov et al., 2021)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDiffusionSteps,
        };
        m.AdditionalInfo["Architecture"] = "GradTTS";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GradTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
