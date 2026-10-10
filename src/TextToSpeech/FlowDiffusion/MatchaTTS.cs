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

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Matcha-TTS: a non-autoregressive acoustic model whose decoder is trained with optimal-transport conditional
/// flow matching and samples a mel spectrogram in a few ODE steps.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching" (Mehta et al.,
/// ICASSP 2024) and its reference implementation (shivammehta25/Matcha-TTS) for what the paper leaves unstated.</para>
/// <para>
/// The encoder is Grad-TTS's (a Glow-TTS text encoder) with rotary position embeddings in place of relative ones: an
/// embedding scaled by √d, a 3-layer convolutional pre-net, 6 post-norm Transformer layers, a projection to per-token
/// mel means μ and a duration predictor on the stop-gradient encoder output. Training aligns μ to the mel by monotonic
/// alignment search. The decoder <c>MatchaDecoder</c> is a 1-D U-Net with Transformer blocks that estimates the vector
/// field v_θ(x_t, μ, t) of the OT-CFM path <c>x_t = (1 − (1 − σ_min) t) x₀ + t x₁</c> with target
/// <c>u = x₁ − (1 − σ_min) x₀</c>, x₀ ~ N(0, I). The loss is the duration loss + the prior loss + the CFM loss on
/// mel spectrograms normalized by the dataset mean and standard deviation; Adam at 1e-4, gradients clipped to norm 5.
/// Synthesis draws x₀ ~ N(0, τ² I) and integrates the field from t = 0 to 1 with Euler steps, then denormalizes.
/// </para>
/// <para>The model aligns itself, so a plain <c>Train(tokens, mel)</c> is its full training signal. It outputs a mel
/// spectrogram; the paper pairs it with a HiFi-GAN vocoder.</para>
/// <para><b>For Beginners:</b> Matcha-TTS learns a straight-line path from random noise to a spectrogram, so turning
/// noise into speech takes only a handful of steps instead of the hundreds diffusion models need.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new MatchaTTS&lt;double&gt;(architecture, "matcha_tts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new MatchaTTS&lt;double&gt;(architecture, new MatchaTTSOptions());
/// var mel = trainModel.TextToMel("hello world");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching",
    "https://arxiv.org/abs/2309.03199",
    Year = 2024,
    Authors = "Mehta et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 32,
                Source = "Mehta et al. 2024, Sec. 4: a learning rate of 1e-4 at a batch size of 32; the reference "
                        + "implementation's optimizer is torch.optim.Adam (configs/model/optimizer/adam.yaml).")]
public partial class MatchaTTS<T> : TtsModelBase<T>, IEndToEndTts<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly MatchaTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;

    // Encoder: Grad-TTS's text encoder with rotary position embeddings.
    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _prenet;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _meanProjection;
    private VariancePredictorLayer<T>? _durationPredictor;

    // Decoder: the U-Net vector field estimator v_theta(x_t, mu, t).
    private MatchaDecoder<T>? _decoder;

    public override ModelOptions GetOptions() => _options;

    public MatchaTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        MatchaTTSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new MatchaTTSOptions();
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

    public MatchaTTS(
        NeuralNetworkArchitecture<T> architecture,
        MatchaTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new MatchaTTSOptions();
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        MaxGradNorm = NumOps.FromDouble(_options.GradientClipNorm);
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
                o.DropoutRate, window: 0, rotary: true));
        _meanProjection = new DenseLayer<T>(o.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _durationPredictor = new VariancePredictorLayer<T>(h, o.DurationPredictorFilterChannels, 1, o.EncoderKernelSize, o.DropoutRate);
        _decoder = new MatchaDecoder<T>(Engine, 2 * o.MelChannels, o.MelChannels,
            Enumerable.Repeat(o.FlowDim, o.DecoderLevels).ToArray(), o.DecoderDropout, o.DecoderHeadDim, o.DecoderHeads,
            o.DecoderMidBlocks);

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_meanProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(_decoder.Layers);
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

    /// <summary>Smallest length the U-Net accepts: a multiple of 4 (the reference's <c>fix_len_compatibility</c>) and of
    /// every down-sampling.</summary>
    private int CompatibleLength(int length)
    {
        int multiple = 1 << Math.Max(2, _options.DecoderLevels - 1);
        return (length + multiple - 1) / multiple * multiple;
    }

    private Tensor<T> FrameMask(int frames, int padded)
    {
        var mask = new Tensor<T>(new[] { padded });
        for (int i = 0; i < frames; i++) mask[i] = NumOps.One;
        return mask;
    }

    private Tensor<T> MaskRows(Tensor<T> mask)
        => Engine.TensorTranspose(Engine.TensorTile(Engine.Reshape(mask, new[] { 1, mask.Length }), new[] { _options.MelChannels, 1 }));

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
    /// <remarks>Inference (reference <c>synthesise</c>): durations <c>⌈exp(log d̂)⌉ · length scale</c>, μ expanded and
    /// padded to a compatible length, <c>x₀ = τ ε</c>, then <see cref="MatchaTTSOptions"/>.NumFlowSteps Euler steps
    /// <c>x ← x + Δt · v_θ(x, μ, t)</c> from t = 0, and the result denormalized with the mel statistics.</remarks>
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
        if (durations.Sum() == 0) durations[durations.Length - 1] = 1;
        int frames = durations.Sum();
        int padded = CompatibleLength(frames);
        var mu = PadFrames(LengthRegulator.Expand(means, durations), padded);
        var mask = FrameMask(frames, padded);

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var x = Gaussian(mu._shape, random, _options.Temperature);
        int steps = _options.NumFlowSteps;
        double dt = 1.0 / steps;
        for (int i = 0; i < steps; i++)
            x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(_decoder!.Estimate(x, mu, i * dt, mask), NumOps.FromDouble(dt)));
        var mel = Engine.TensorSlice(x, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
        return Engine.TensorAddScalar(Engine.TensorMultiplyScalar(mel, NumOps.FromDouble(_options.MelStd)), NumOps.FromDouble(_options.MelMean));
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
    /// <remarks>Matcha-TTS finds its own alignment with monotonic alignment search, so a token/mel pair is its whole supervision.</remarks>
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

    /// <remarks>The CFM loss is an expectation over t and the noise; evaluation fixes them with the sampling seed so the
    /// same parameters always score the same.</remarks>
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

    private sealed record TrainingDraw(double Time, Tensor<T> Noise);

    // t ~ U(0, 1) and x0 ~ N(0, I) over the padded utterance.
    private TrainingDraw DrawTrainingRandomness(int frames, Random random)
    {
        double t = random.NextDouble();
        return new TrainingDraw(t, Gaussian(new[] { CompatibleLength(frames), _options.MelChannels }, random, 1.0));
    }

    /// <summary>
    /// The training objective (reference <c>MatchaTTS.forward</c> and <c>CFM.compute_loss</c>) on the normalized mel y,
    /// zero-padded to a compatible length: the duration loss <c>Σ (log d̂ − log(1e-8 + d))² / tokens</c> against the
    /// MAS durations; the prior loss <c>Σ ½((y − μ)² + log 2π) · mask / (frames · mel)</c>; and the CFM loss
    /// <c>Σ (v_θ(x_t, μ, t) − u)² / (frames · mel)</c> with <c>x_t = (1 − (1 − σ_min) t) x₀ + t y</c> and
    /// <c>u = y − (1 − σ_min) x₀</c>.
    /// </summary>
    /// <remarks>As in the reference, <c>u</c> is not masked: on padded frames the estimate is 0 and u is the scaled
    /// noise, a parameter-independent constant in the loss.</remarks>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel, TrainingDraw draw)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var normalized = Engine.TensorMultiplyScalar(Engine.TensorAddScalar(mel, NumOps.FromDouble(-_options.MelMean)),
            NumOps.FromDouble(1.0 / _options.MelStd));
        var (hidden, means) = Encode(tokens);

        int tokenCount = means.Shape[0];
        var scores = new double[tokenCount, frames];
        for (int i = 0; i < tokenCount; i++)
            for (int j = 0; j < frames; j++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    double d = NumOps.ToDouble(normalized[j, c]) - NumOps.ToDouble(means[i, c]);
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

        int padded = draw.Noise.Shape[0];
        var y = PadFrames(normalized, padded);
        var mu = PadFrames(LengthRegulator.Expand(means, durations), padded);
        var mask = FrameMask(frames, padded);
        var maskRows = MaskRows(mask);
        double count = frames * (double)channels;

        var priorDiff = Engine.TensorSubtract(y, mu);
        var priorSum = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(priorDiff, priorDiff), maskRows), new[] { 0, 1 }, keepDims: false);
        var priorLoss = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(priorSum, NumOps.FromDouble(0.5 / count)),
            NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI)));

        double t = draw.Time, sigma = _options.SigmaMin;
        var xt = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(draw.Noise, NumOps.FromDouble(1.0 - (1.0 - sigma) * t)),
            Engine.TensorMultiplyScalar(y, NumOps.FromDouble(t)));
        var u = Engine.TensorSubtract(y, Engine.TensorMultiplyScalar(draw.Noise, NumOps.FromDouble(1.0 - sigma)));
        var residual = Engine.TensorSubtract(_decoder!.Estimate(xt, mu, t, mask), u);
        var flowLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(residual, residual), new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(1.0 / count));

        return Engine.TensorAdd(Engine.TensorAdd(durationLoss, priorLoss), flowLoss);
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
            Name = _useNativeMode ? "Matcha-TTS-Native" : "Matcha-TTS-ONNX",
            Description = "Matcha-TTS: OT-CFM Fast TTS (Mehta et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "MatchaTTS";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate;
        m.AdditionalInfo["MelChannels"] = _options.MelChannels;
        m.AdditionalInfo["HopSize"] = _options.HopSize;
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MatchaTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
