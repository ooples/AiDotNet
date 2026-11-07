using AiDotNet.Attributes;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// DOVE: efficient one-step diffusion for real-world video super-resolution, fine-tuned from CogVideoX1.5-5B.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DOVE (Chen et al., NeurIPS 2025) adds no module to CogVideoX1.5: the clip's bilinear x4 upscale is encoded by the causal
/// 3D VAE and taken as the noisy latent at timestep 399, and one v-prediction step of the expert transformer,
/// z_sr = sqrt(ᾱ_t)·z_lr − sqrt(1 − ᾱ_t)·v_θ(z_lr, c, t), gives the latent the VAE decodes (Eq. 2). The prompt is empty and
/// pre-encoded. Only the transformer trains: first on the MSE between z_sr and the high-resolution latent (Eq. 3), then in
/// pixel space with the VAE run frame by frame, on MSE, DISTS and the frame-difference loss, mostly on single frames
/// (Eqs. 4-7). The release (zhengchen1999/DOVE) fixes the remaining details: frames padded to 8N + 1 by repeating the
/// last, sides to multiples of 16, the posterior sampled, and the first latent frame repeated to fill the temporal patch.
/// </para>
/// <para>
/// <b>For Beginners:</b> DOVE enlarges each frame the simple way and then lets a large video generation model clean the
/// enlargement up in a single pass. Running one step instead of dozens makes it many times faster than other diffusion
/// upscalers.
/// </para>
/// <para>
/// The empty prompt's T5-XXL embedding is part of the checkpoint (the release ships it pre-encoded) and is held as a frozen
/// tensor; untrained, it is zero.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2505.16239",
    Year = 2025,
    Authors = "Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, Yulun Zhang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = 1e-4,
                Component = "adaptation", Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 100,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Constant,
                Source = "Chen et al. 2025, Sec. 4.1: AdamW (0.9, 0.95), 10k latent-space iterations at 2e-5. The release "
                        + "(finetune/schemas/args.py, train_ddp_one_s1.sh) adds weight decay 1e-4, a 100-step linear warmup "
                        + "and a 0.1 gradient clip. The paper's beta3 0.98 is the release's CAME-only argument.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-6, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = 1e-4,
                Component = "refinement", Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 100,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Constant,
                Source = "Chen et al. 2025, Sec. 4.1: 500 pixel-space iterations at 5e-6, otherwise as the first stage.")]
public partial class DOVE<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly DOVEOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _adaptationOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _refinementOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private CogVideoXTransformer<T>? _transformer;
    private int _promptEmbedding;
    private CausalVideoEncoder<T>? _vaeEncoder;
    private CausalVideoDecoder<T>? _vaeDecoder;
    private DistsBlock<T>? _dists;
    private double[] _alphasCumprod = Array.Empty<double>();
    private Random _random;

    /// <summary>Creates a DOVE model in ONNX inference mode.</summary>
    public DOVE(NeuralNetworkArchitecture<T> architecture, string modelPath, DOVEOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new DOVEOptions() : new DOVEOptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DOVE model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's and CogVideoX1.5-5B's defaults when null.</param>
    /// <param name="optimizer">The adaptation stage's optimizer; the release's AdamW when null. Refinement always uses AdamW
    /// at <see cref="DOVEOptions.RefinementLearningRate"/>.</param>
    public DOVE(NeuralNetworkArchitecture<T> architecture, DOVEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _adaptationOptimizer = optimizer;
        _options = options is null ? new DOVEOptions() : new DOVEOptions(options);
        Options = _options;
        var o = _options;
        if (o.NumAttentionHeads < 1 || o.AttentionHeadDim < 16 || o.AttentionHeadDim % 16 != 0 || o.NumLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The transformer needs heads, a head width that is a multiple of 16, and layers.");
        if (o.VaeChannels is not { Length: >= 1 } || o.VaeChannels.Any(c => c < 32 || c % 32 != 0))
            throw new ArgumentOutOfRangeException(nameof(options), "VaeChannels must be multiples of 32.");
        if (o.TemporalCompressionRatio < 1 || (o.TemporalCompressionRatio & (o.TemporalCompressionRatio - 1)) != 0
            || Log2(o.TemporalCompressionRatio) > o.VaeChannels.Length - 1)
            throw new ArgumentOutOfRangeException(nameof(options), "TemporalCompressionRatio must be a power of two the VAE's stages can reach.");
        if (o.ScaleFactor < 1 || o.PatchSize < 1 || o.PatchSizeT < 1 || o.Timestep < 0 || o.Timestep >= o.NumTrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor, the patch sizes and the timestep must be valid.");
        _useNativeMode = true;
        ScaleFactor = o.ScaleFactor;
        NumFrames = o.NumFrames;
        _random = o.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        // The step count selects the training stage, so a clone continues in the stage the original was in.
        state.DeclareDoubleVector(nameof(DOVE<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
            restored => _trainingSteps = restored is { Length: > 0 } ? (int)restored[0] : 0);
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        // Sec. 3.3: only the transformer is fine-tuned; the VAE stays frozen to keep its priors.
        RegisterParameterComponent("DOVE.VaeEncoder", _vaeEncoder, AiDotNet.Models.Parameters.ParameterSlotRole.Frozen);
        RegisterParameterComponent("DOVE.VaeDecoder", _vaeDecoder, AiDotNet.Models.Parameters.ParameterSlotRole.Frozen);
    }

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #region Layers and network

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }
        var o = _options;
        _transformer = new CogVideoXTransformer<T>(Layers, o.LatentChannels, o.NumAttentionHeads, o.AttentionHeadDim, o.NumLayers,
            o.TimeEmbedDim, o.PatchSize, o.PatchSizeT, o.NormEpsilon);
        _promptEmbedding = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.MaxTextSeqLength * o.TextEmbedDim, 0.0));
        int level = Log2(o.TemporalCompressionRatio);
        _vaeEncoder = new CausalVideoEncoder<T>(3, o.LatentChannels, o.VaeChannels, o.VaeLayersPerBlock, 32, 1e-6, level);
        _vaeDecoder = new CausalVideoDecoder<T>(o.LatentChannels, 3, o.VaeChannels, o.VaeLayersPerBlock, 32, 1e-6, level);
        // A zero DISTS weight leaves the VGG16 out entirely rather than carrying it unused.
        if (o.DistsWeight > 0) _dists = new DistsBlock<T>(Layers);
        _alphasCumprod = AlphasCumprod(o);
    }

    private static int Log2(int value)
    {
        int log = 0;
        while ((1 << log) < value) log++;
        return log;
    }

    private int SpatialFactor => 1 << (_options.VaeChannels.Length - 1);

    // The low-resolution side unit the release pads to: the VAE's factor times the patch (16 at the defaults).
    private int SideUnit => SpatialFactor * _options.PatchSize;

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        var output = RunClips(new Tensor<T>(new[] { 1, 1, 3, SideUnit, SideUnit }), InferenceRandom());
        var frames = Engine.Reshape(output, new[] { 1, 3, output.Shape[3], output.Shape[4] });
        _dists?.Loss(Engine, Layers, frames, frames);
        MarkLayerShapesResolved();
    }

    private CogVideoXTransformer<T> Transformer => _transformer ?? throw new InvalidOperationException("DOVE's native layers are not initialized.");

    private CausalVideoEncoder<T> VaeEncoder => _vaeEncoder ?? throw new InvalidOperationException("DOVE's native layers are not initialized.");

    private CausalVideoDecoder<T> VaeDecoder => _vaeDecoder ?? throw new InvalidOperationException("DOVE's native layers are not initialized.");

    /// <summary>The transformer's parameters, the only ones DOVE trains.</summary>
    internal IReadOnlyList<Tensor<T>> TransformerParameters => _usingCustomLayers || _transformer is null
        ? Array.Empty<Tensor<T>>()
        : TapeTrainingStep<T>.CollectParameters(Layers.Skip(_transformer.First).Take(_transformer.End - _transformer.First));

    /// <summary>The causal VAE's weights.</summary>
    internal IReadOnlyList<Tensor<T>> VaeParameters => _vaeEncoder is null || _vaeDecoder is null
        ? Array.Empty<Tensor<T>>()
        : _vaeEncoder.NamedTensors("").Concat(_vaeDecoder.NamedTensors("")).Select(named => named.Tensor).ToList();

    /// <summary>The training stage the next step runs: 0 latent-space adaptation, 1 pixel-space refinement.</summary>
    internal int TrainingStage => _trainingSteps < _options.AdaptationIterations ? 0 : 1;

    #endregion

    #region One-step diffusion

    /// <summary>
    /// CogVideoXDDIMScheduler's ᾱ: the scaled-linear β (sqrt-linear from BetaStart to BetaEnd, squared), rescaled to zero
    /// terminal SNR (Lin et al. 2024) as the CogVideoX 1.5 scheduler config asks.
    /// </summary>
    internal static double[] AlphasCumprod(DOVEOptions options)
    {
        int n = options.NumTrainTimesteps;
        var alphaBarSqrt = new double[n];
        double product = 1.0;
        for (int i = 0; i < n; i++)
        {
            double root = Math.Sqrt(options.BetaStart) + (n == 1 ? 0 : i / (double)(n - 1)) * (Math.Sqrt(options.BetaEnd) - Math.Sqrt(options.BetaStart));
            product *= 1.0 - root * root;
            alphaBarSqrt[i] = Math.Sqrt(product);
        }
        double first = alphaBarSqrt[0], last = alphaBarSqrt[n - 1];
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            double shifted = (alphaBarSqrt[i] - last) * first / (first - last);
            result[i] = shifted * shifted;
        }
        return result;
    }

    // Inference samples the posterior from a fixed seed so the same clip always upscales to the same video.
    private Random InferenceRandom() => RandomHelper.CreateSeededRandom(_options.Seed ?? 42);

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return t;
    }

    private Tensor<T> ToSigned(Tensor<T> x) => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));

    // The decoder's [-1, 1] output to [0, 1], clamped as the release does.
    private Tensor<T> ToUnit(Tensor<T> x)
        => Engine.TensorClamp(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(x, NumOps.One), NumOps.FromDouble(0.5)), NumOps.Zero, NumOps.One);

    // Repeats the last frame of [B, F, ...] until (F − 1) is a multiple of 2·temporal compression (the release's 8N + 1).
    private Tensor<T> PadFrames(Tensor<T> clips)
    {
        int f = clips.Shape[1], unit = 2 * _options.TemporalCompressionRatio, remainder = (f - 1) % unit;
        if (remainder == 0) return clips;
        var last = Engine.TensorNarrow(clips, 1, f - 1, 1);
        var target = clips.Shape.ToArray();
        target[1] = unit - remainder;
        return Engine.TensorConcatenate(new[] { clips, Engine.TensorBroadcastTo(last, target) }, 1);
    }

    // Zero-pads the right and bottom of [B, F, 3, h, w] to multiples of unit.
    private Tensor<T> PadSides(Tensor<T> clips, int unit)
    {
        int h = clips.Shape[3], w = clips.Shape[4];
        int padH = (unit - h % unit) % unit, padW = (unit - w % unit) % unit;
        if (padH == 0 && padW == 0) return clips;
        return Engine.PadNd(clips, new[] { 0, padW, 0, padH }, PadMode.Constant, NumOps.Zero);
    }

    // Low-resolution clips [B, F, 3, h, w] in [0, 1] to the bilinear upscale [B, 3, F, s·h, s·w] in [-1, 1].
    private Tensor<T> Upsample(Tensor<T> clips)
    {
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var up = Engine.Interpolate(Engine.Reshape(clips, new[] { b * f, 3, h, w }), new[] { h * s, w * s }, InterpolateMode.Bilinear, false);
        return Engine.TensorPermute(Engine.Reshape(ToSigned(up), new[] { b, f, 3, h * s, w * s }), new[] { 0, 2, 1, 3, 4 });
    }

    // Frames [B, F, 3, H, W] in [0, 1] to the VAE's layout [B, 3, F, H, W] in [-1, 1].
    private Tensor<T> ToVideo(Tensor<T> clips) => Engine.TensorPermute(ToSigned(clips), new[] { 0, 2, 1, 3, 4 });

    // A posterior sample of the whole clip, scaled: [B, 3, F, H, W] to [B, C, 1 + (F − 1) / r, H / f, W / f].
    private Tensor<T> EncodeVideo(Tensor<T> video, Random random)
    {
        var moments = VaeEncoder.Forward(video);
        int c = _options.LatentChannels;
        var mean = Engine.TensorNarrow(moments, 1, 0, c);
        var logVar = Engine.TensorClamp(Engine.TensorNarrow(moments, 1, c, c), NumOps.FromDouble(-30.0), NumOps.FromDouble(20.0));
        var std = Engine.TensorExp(Engine.TensorMultiplyScalar(logVar, NumOps.FromDouble(0.5)));
        var sample = Engine.TensorAdd(mean, Engine.TensorMultiply(std, Gaussian(mean.Shape.ToArray(), random)));
        return Engine.TensorMultiplyScalar(sample, NumOps.FromDouble(_options.LatentScale));
    }

    // The refinement stage's frame-by-frame encoding: one latent frame per frame.
    private Tensor<T> EncodeFrames(Tensor<T> video, Random random)
    {
        int f = video.Shape[2];
        var latents = new Tensor<T>[f];
        for (int i = 0; i < f; i++) latents[i] = EncodeVideo(Engine.TensorNarrow(video, 2, i, 1), random);
        return Engine.TensorConcatenate(latents, 2);
    }

    private Tensor<T> DecodeVideo(Tensor<T> latents)
        => VaeDecoder.Forward(Engine.TensorMultiplyScalar(latents, NumOps.FromDouble(1.0 / _options.LatentScale)));

    private Tensor<T> DecodeFrames(Tensor<T> latents)
    {
        int f = latents.Shape[2];
        var frames = new Tensor<T>[f];
        for (int i = 0; i < f; i++) frames[i] = DecodeVideo(Engine.TensorNarrow(latents, 2, i, 1));
        return Engine.TensorConcatenate(frames, 2);
    }

    /// <summary>
    /// The one step on latents [B, C, F, h, w]: the first frame repeated F mod p_t times in front (as the release does),
    /// then z_sr = sqrt(ᾱ_t)·z − sqrt(1 − ᾱ_t)·v_θ(z, c, t).
    /// </summary>
    /// <returns>z_sr with the repeated frames still in front, and their count.</returns>
    private (Tensor<T> Latents, int Copies) OneStep(Tensor<T> z)
    {
        int b = z.Shape[0], copies = z.Shape[2] % _options.PatchSizeT;
        if (copies > 0)
        {
            var first = Engine.TensorNarrow(z, 2, 0, 1);
            var shape = z.Shape.ToArray();
            shape[2] = copies;
            z = Engine.TensorConcatenate(new[] { Engine.TensorBroadcastTo(first, shape), z }, 2);
        }
        var frames = Engine.TensorPermute(z, new[] { 0, 2, 1, 3, 4 });
        var prompt = Engine.Reshape(RealViformerBlocks.Read(Layers, _promptEmbedding), new[] { 1, _options.MaxTextSeqLength, _options.TextEmbedDim });
        prompt = Engine.TensorBroadcastTo(prompt, new[] { b, _options.MaxTextSeqLength, _options.TextEmbedDim });
        var v = Transformer.Forward(Engine, Layers, frames, prompt, Enumerable.Repeat((double)_options.Timestep, b).ToArray());
        double alphaBar = _alphasCumprod[_options.Timestep];
        var output = Engine.TensorSubtract(Engine.TensorMultiplyScalar(frames, NumOps.FromDouble(Math.Sqrt(alphaBar))),
            Engine.TensorMultiplyScalar(v, NumOps.FromDouble(Math.Sqrt(1.0 - alphaBar))));
        return (Engine.TensorPermute(output, new[] { 0, 2, 1, 3, 4 }), copies);
    }

    private Tensor<T> DropCopies(Tensor<T> latents, int copies)
        => copies == 0 ? latents : Engine.TensorNarrow(latents, 2, copies, latents.Shape[2] - copies);

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the output size, in [0, 1].
    private Tensor<T> RunClips(Tensor<T> frames, Random random)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var low = PadSides(PadFrames(clips), SideUnit);
        var (latents, copies) = OneStep(EncodeVideo(Upsample(low), random));
        var video = ToUnit(DecodeVideo(DropCopies(latents, copies)));
        video = Engine.TensorSlice(video, new[] { 0, 0, 0, 0, 0 }, new[] { b, 3, f, h * s, w * s });
        var output = Engine.TensorPermute(video, new[] { 0, 2, 1, 3, 4 });
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        using var noGrad = new NoGradScope<T>();
        return RunClips(input, InferenceRandom());
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input), InferenceRandom());

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> ReleaseAdamW(double rate)
        => new(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate,
            Beta1 = 0.9,
            Beta2 = 0.95,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = _options.MaxGradientNorm > 0,
            MaxGradientNorm = _options.MaxGradientNorm > 0 ? _options.MaxGradientNorm : 1.0,
            LearningRateScheduler = new LinearWarmupScheduler(rate, Math.Max(0, _options.WarmupSteps), 0, 0.0,
                LinearWarmupScheduler.DecayMode.Constant),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> AdaptationOptimizer()
        => _adaptationOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdamW(_options.LearningRate), "adaptation");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> RefinementOptimizer()
        => _refinementOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdamW(_options.RefinementLearningRate), "refinement");

    private Tensor<T> Mse(Tensor<T> a, Tensor<T> b)
    {
        var squared = Engine.TensorSquare(Engine.TensorSubtract(a, b));
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    // Input and target as clips [B, F, 3, h, w] and [B, F, 3, H, W] in [0, 1].
    private (Tensor<T> Low, Tensor<T> High) AsClips(Tensor<T> input, Tensor<T> target)
    {
        var low = PreprocessFrames(input);
        var high = NormalizeFrames(target);
        return low.Rank == 5
            ? (low, high)
            : (Engine.Reshape(low, new[] { 1 }.Concat(low._shape).ToArray()), Engine.Reshape(high, new[] { 1 }.Concat(high._shape).ToArray()));
    }

    // The adaptation stage's latents (no gradient: the VAE is frozen): the clip padded as at inference, both encoded whole.
    private (Tensor<T> Low, Tensor<T> High) AdaptationLatents(Tensor<T> low, Tensor<T> high, Random random)
    {
        using var noGrad = new NoGradScope<T>();
        var lowLatent = EncodeVideo(Upsample(PadSides(PadFrames(low), SideUnit)), random);
        var highLatent = EncodeVideo(ToVideo(PadSides(PadFrames(high), SideUnit * _options.ScaleFactor)), random);
        return (lowLatent, highLatent);
    }

    // Eq. 3 with the release's padding: the MSE of z_sr against the high-resolution latent, the repeated frames included.
    private Tensor<T> AdaptationLoss(Tensor<T> lowLatent, Tensor<T> highLatent)
    {
        var (prediction, copies) = OneStep(lowLatent);
        if (copies > 0)
        {
            var shape = highLatent.Shape.ToArray();
            shape[2] = copies;
            highLatent = Engine.TensorConcatenate(new[] { Engine.TensorBroadcastTo(Engine.TensorNarrow(highLatent, 2, 0, 1), shape), highLatent }, 2);
        }
        return Mse(prediction, highLatent);
    }

    // Eqs. 4-7: frame-by-frame VAE, MSE, DISTS per frame and the L1 of consecutive-frame differences.
    private Tensor<T> RefinementLoss(Tensor<T> lowLatent, Tensor<T> high)
    {
        var (prediction, copies) = OneStep(lowLatent);
        var video = ToUnit(DecodeFrames(DropCopies(prediction, copies)));
        var target = Engine.TensorPermute(high, new[] { 0, 2, 1, 3, 4 });
        var loss = Mse(video, target);
        int b = video.Shape[0], f = video.Shape[2], h = video.Shape[3], w = video.Shape[4];
        if (_dists is not null)
        {
            Tensor<T> Frames(Tensor<T> v) => Engine.Reshape(Engine.TensorPermute(v, new[] { 0, 2, 1, 3, 4 }), new[] { b * f, 3, h, w });
            loss = Engine.TensorAdd(loss, Weighted(_dists.Loss(Engine, Layers, Frames(video), Frames(target)), _options.DistsWeight));
        }
        if (f > 1 && _options.FrameDifferenceWeight > 0)
        {
            var generated = Engine.TensorSubtract(Engine.TensorNarrow(video, 2, 1, f - 1), Engine.TensorNarrow(video, 2, 0, f - 1));
            var truth = Engine.TensorSubtract(Engine.TensorNarrow(target, 2, 1, f - 1), Engine.TensorNarrow(target, 2, 0, f - 1));
            var l1 = Engine.TensorAbs(Engine.TensorSubtract(generated, truth));
            loss = Engine.TensorAdd(loss, Weighted(Engine.ReduceMean(l1, Enumerable.Range(0, l1.Rank).ToArray(), false), _options.FrameDifferenceWeight));
        }
        return loss;
    }

    private Tensor<T> RefinementLatents(Tensor<T> low, Random random)
    {
        using var noGrad = new NoGradScope<T>();
        return EncodeFrames(Upsample(low), random);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The paper's two stages in turn, training only the transformer: latent-space adaptation, then pixel-space refinement,
    /// where a share <see cref="DOVEOptions.ImageRatio"/> of the steps trains on one frame of the clip (the paper's image
    /// branch on DIV2K stills).
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        int stage = TrainingStage;
        _trainingSteps++;
        var (low, high) = AsClips(input, expectedOutput);
        if (stage == 0)
        {
            var (lowLatent, highLatent) = AdaptationLatents(low, high, _random);
            return new[]
            {
                new TrainingObjective<T>(new[] { input, expectedOutput }, _ => AdaptationLoss(lowLatent, highLatent))
                {
                    Optimizer = AdaptationOptimizer(),
                    Parameters = () => TransformerParameters,
                },
            };
        }

        if (_random.NextDouble() < _options.ImageRatio)
        {
            int frame = _random.Next(low.Shape[1]);
            low = Engine.TensorNarrow(low, 1, frame, 1);
            high = Engine.TensorNarrow(high, 1, frame, 1);
        }
        var latents = RefinementLatents(low, _random);
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, _ => RefinementLoss(latents, high))
            {
                Optimizer = RefinementOptimizer(),
                Parameters = () => TransformerParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.DiffusionDenoising;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The current stage's loss on the whole clip with the posterior sampled from a fixed seed.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var (low, high) = AsClips(input, target);
        var random = RandomHelper.CreateSeededRandom(1729);
        if (TrainingStage == 0)
        {
            var (lowLatent, highLatent) = AdaptationLatents(low, high, random);
            return AdaptationLoss(lowLatent, highLatent)[0];
        }
        return RefinementLoss(RefinementLatents(low, random), high)[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? AdaptationOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <summary>Frames are pixels on the 0-255 scale; the network divides them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>Upscaled frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "DOVE-Native" : "DOVE-ONNX",
            Description = "DOVE one-step diffusion video super-resolution on CogVideoX1.5 (Chen et al. 2025)",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumLayers"] = _options.NumLayers.ToString();
        m.AdditionalInfo["Timestep"] = _options.Timestep.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DOVE<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
