using AiDotNet.Attributes;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
/// MGLD-VSR: motion-guided latent diffusion for temporally consistent real-world video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MGLD-VSR (Yang et al., ECCV 2024) builds on StableSR: a frozen Stable Diffusion 2.1 UNet whose residual blocks are
/// SPADE-modulated by a time-aware encoding of the LR latent, with temporal convolutions and temporal attention added to
/// its middle block over five-frame clips. Sampling starts from the LR latent noised to the last timestep and, after
/// every DDPM step, moves the latents down the gradient of an occlusion-masked warping error between neighbouring
/// latents (flows from RAFT on the LR frames). A video decoder (SD's KL decoder with CFW modules fusing the LQ encoder's
/// features and temporal convolutions) is trained afterwards with L1, LPIPS, frame-difference, edge-weighted warping and
/// adversarial losses.
/// </para>
/// <para>
/// <b>For Beginners:</b> Diffusion models make images by removing noise step by step. MGLD-VSR uses optical flow to keep
/// neighbouring frames consistent while it removes the noise, and a decoder that also looks at neighbouring frames, so
/// the upscaled video does not flicker.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution",
    "https://arxiv.org/abs/2312.00853",
    Year = 2024,
    Authors = "Xi Yang, Chenhang He, Jianqi Ma, Lei Zhang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Component = "diffusion",
                Source = "The release (mgldvsr_512_realbasicvsr_deg.yaml, LatentDiffusionVSRTextWT): AdamW at the base learning "
                        + "rate 5e-5 over SPADE, the temporal modules and the structure encoder.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, Beta1 = 0.5, Beta2 = 0.9, Component = "decoder",
                Source = "The release (video_autoencoder_kl_64x64x4_resi.yaml, VideoAutoencoderKLResi): Adam at 5e-5 with "
                        + "betas (0.5, 0.9) over the CFW modules and the decoder's temporal convolutions.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, Beta1 = 0.5, Beta2 = 0.9, Component = "discriminator",
                Source = "The release: the same Adam for the PatchGAN discriminator.")]
public partial class MGLDVSR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly MGLDVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _diffusionOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _decoderOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private int _contextLayer;
    private SpyNetFlowBlock<T>? _spyNet;
    private LpipsVggBlock<T>? _lpips;
    private readonly LazyModuleGroup<T> _vae = new(MgldInit.Default);
    private readonly MgldParts<T> _diffusion = new();
    private readonly MgldParts<T> _decoderParts = new();
    private readonly LazyModuleGroup<T> _raftGroup = new(MgldInit.Default);
    private KlEncoder<T>? _encoder;
    private MgldConv2d<T>? _quantConv;
    private MgldUNet<T>? _unet;
    private MgldStructureEncoder<T>? _structure;
    private MgldVideoDecoder<T>? _decoder;
    private MgldDiscriminator<T>? _discriminator;
    private MgldRaft<T>? _raft;
    private double[] _alphasCumprod = Array.Empty<double>();
    private Random _random;

    /// <summary>Creates an MGLD-VSR model in ONNX inference mode.</summary>
    public MGLDVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, MGLDVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new MGLDVSROptions() : new MGLDVSROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an MGLD-VSR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's and SD 2.1's defaults when null.</param>
    /// <param name="optimizer">The diffusion stage's optimizer; the release's AdamW when null.</param>
    public MGLDVSR(NeuralNetworkArchitecture<T> architecture, MGLDVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _diffusionOptimizer = optimizer;
        _options = options is null ? new MGLDVSROptions() : new MGLDVSROptions(options);
        Options = _options;
        var o = _options;
        bool Grouped(int width) => width > 0 && width % 32 == 0;
        if (o.UNetMultipliers is not { Length: > 0 } || o.UNetMultipliers.Any(m => !Grouped(m * o.UNetChannels)) || o.UNetResBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Every UNet width must split into 32 groups.");
        if (o.HeadChannels < 1 || o.UNetMultipliers.Any(m => m * o.UNetChannels % o.HeadChannels != 0) || o.AttentionLevels < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Every UNet width must split into HeadChannels-wide heads.");
        if (o.StructureMultipliers is not { Length: > 0 } || o.StructureMultipliers.Length != o.UNetMultipliers.Length
            || o.StructureMultipliers.Any(m => !Grouped(m * o.StructureChannels)) || o.StructureResBlocks < 1 || o.StructureHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The structure encoder needs one 32-group level per UNet level.");
        if (o.VaeMultipliers is not { Length: >= 3 } || o.VaeMultipliers.Any(m => !Grouped(m * o.VaeChannels)) || o.VaeResBlocks < 1 || o.FuseBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The autoencoder needs at least three 32-group levels.");
        if (o.ContextDim < 1 || o.ContextLength < 1 || o.NumFrames < 1 || o.ScaleFactor < 1 || o.NumTrainTimesteps < 2
            || o.SamplingSteps < 1 || o.SamplingSteps > o.NumTrainTimesteps || o.FlowIterations < 1 || o.MinimumSide < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The context, clip length, schedule and sampling must be valid.");
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
        state.DeclareDoubleVector(nameof(MGLDVSR<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
            restored => _trainingSteps = restored is { Length: > 0 } ? (int)restored[0] : 0);
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        var frozen = AiDotNet.Models.Parameters.ParameterSlotRole.Frozen;
        // Stable Diffusion's autoencoder, UNet and KL decoder and the pretrained RAFT stay as loaded; MGLD-VSR trains
        // SPADE, the temporal modules and the structure encoder, then the CFW modules, the decoder's temporal
        // convolutions and the discriminator.
        RegisterParameterComponent("MGLDVSR.Vae", _vae, frozen);
        RegisterParameterComponent("MGLDVSR.UNet", _diffusion.Frozen, frozen);
        RegisterParameterComponent("MGLDVSR.DiffusionAdapters", _diffusion.Trained);
        RegisterParameterComponent("MGLDVSR.Decoder", _decoderParts.Frozen, frozen);
        RegisterParameterComponent("MGLDVSR.DecoderAdapters", _decoderParts.Trained);
        RegisterParameterComponent("MGLDVSR.Discriminator", _discriminator);
        RegisterParameterComponent("MGLDVSR.Raft", _raftGroup, frozen);
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
        // The empty prompt's OpenCLIP embedding: a constant the release computes once and never trains.
        _contextLayer = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.ContextLength * o.ContextDim, 0.0));
        var vaeWidths = o.VaeMultipliers.Select(m => m * o.VaeChannels).ToArray();
        _encoder = _vae.Add(new KlEncoder<T>(3, Latent, vaeWidths, o.VaeResBlocks, 32));
        _quantConv = _vae.Add(new MgldConv2d<T>(2 * Latent, 2 * Latent, 1));
        _structure = new MgldStructureEncoder<T>(_diffusion, Latent, o.StructureChannels, o.StructureChannels, o.StructureMultipliers,
            o.StructureResBlocks, o.AttentionLevels, o.StructureHeads);
        _unet = new MgldUNet<T>(_diffusion, Latent, o.UNetChannels, o.UNetMultipliers, o.UNetResBlocks, o.AttentionLevels,
            o.HeadChannels, o.ContextDim, o.StructureChannels);
        _decoder = new MgldVideoDecoder<T>(_decoderParts, Latent, o.VaeChannels, o.VaeMultipliers, o.VaeResBlocks, o.FuseBlocks);
        _discriminator = new MgldDiscriminator<T>();
        _raft = new MgldRaft<T>(_raftGroup);
        // The warping loss's flows come from SPyNet (the loss config's flownet); a zero weight leaves it out.
        if (o.TemporalWeight > 0) _spyNet = new SpyNetFlowBlock<T>(Layers);
        // A zero LPIPS weight leaves the VGG16 out entirely rather than carrying it unused.
        if (o.PerceptualWeight > 0) _lpips = new LpipsVggBlock<T>(Layers);
        _alphasCumprod = AlphasCumprod(o);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        var image = new Tensor<T>(new[] { 1, 3, 32, 32 });
        _lpips?.Loss(Engine, Layers, image, image);
        _spyNet?.Flow(Engine, Layers, image, image);
        MarkLayerShapesResolved();
    }

    private const int Latent = 4;

    private static InvalidOperationException NotInitialized() => new("MGLD-VSR's native layers are not initialized.");

    private MgldUNet<T> UNet => _unet ?? throw NotInitialized();

    private MgldStructureEncoder<T> Structure => _structure ?? throw NotInitialized();

    private MgldVideoDecoder<T> Decoder => _decoder ?? throw NotInitialized();

    private MgldDiscriminator<T> Discriminator => _discriminator ?? throw NotInitialized();

    private MgldRaft<T> Raft => _raft ?? throw NotInitialized();

    private static IReadOnlyList<Tensor<T>> Chunks(AiDotNet.Models.Parameters.IParameterChunkSource<T>? source)
        => source is null ? Array.Empty<Tensor<T>>() : source.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    /// <summary>The diffusion stage's trained weights: SPADE, the temporal modules and the structure encoder.</summary>
    internal IReadOnlyList<Tensor<T>> DiffusionParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : Chunks(_diffusion.Trained);

    /// <summary>The frozen Stable Diffusion UNet's weights.</summary>
    internal IReadOnlyList<Tensor<T>> FrozenUNetParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : Chunks(_diffusion.Frozen);

    /// <summary>The decoder stage's trained weights: the CFW modules and the decoder's temporal convolutions.</summary>
    internal IReadOnlyList<Tensor<T>> DecoderParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : Chunks(_decoderParts.Trained);

    /// <summary>The PatchGAN discriminator's weights.</summary>
    internal IReadOnlyList<Tensor<T>> DiscriminatorParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : Chunks(_discriminator);

    /// <summary>The training stage the next step runs: 0 the diffusion, 1 the video decoder.</summary>
    internal int TrainingStage => _trainingSteps < _options.DiffusionIterations ? 0 : 1;

    #endregion

    #region Schedule

    /// <summary>
    /// ᾱ_t of LDM's "linear" schedule: β spaced linearly in √β from √β_start to √β_end over the training timesteps.
    /// </summary>
    internal static double[] AlphasCumprod(MGLDVSROptions options)
    {
        int n = options.NumTrainTimesteps;
        double a = Math.Sqrt(options.BetaStart), b = Math.Sqrt(options.BetaEnd), product = 1.0;
        var cumprod = new double[n];
        for (int i = 0; i < n; i++)
        {
            double root = a + (b - a) * i / (n - 1);
            product *= 1.0 - root * root;
            cumprod[i] = product;
        }
        return cumprod;
    }

    /// <summary>
    /// <c>space_timesteps(N, [S])</c>: S timesteps spread evenly over [0, N − 1] by rounding a stride of (N − 1)/(S − 1).
    /// </summary>
    internal static int[] SpacedTimesteps(int total, int steps)
    {
        if (steps == 1) return new[] { 0 };
        double stride = (total - 1) / (double)(steps - 1), current = 0;
        var taken = new SortedSet<int>();
        for (int i = 0; i < steps; i++)
        {
            taken.Add((int)Math.Round(current, MidpointRounding.AwayFromZero));
            current += stride;
        }
        return taken.ToArray();
    }

    #endregion

    #region Pipeline

    // Inference draws its noise from a fixed seed so the same clip always upscales to the same video.
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

    private Tensor<T> ToUnit(Tensor<T> x)
        => Engine.TensorClamp(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(x, NumOps.One), NumOps.FromDouble(0.5)), NumOps.Zero, NumOps.One);

    private Tensor<T> Context(int frames)
    {
        var context = Engine.Reshape(RealViformerBlocks.Read(Layers, _contextLayer), new[] { 1, _options.ContextLength, _options.ContextDim });
        return Engine.TensorBroadcastTo(context, new[] { frames, _options.ContextLength, _options.ContextDim });
    }

    // The image side every frame is padded to a multiple of: the autoencoder's x8 times the UNet's halvings.
    private int Unit => 8 << (_options.UNetMultipliers.Length - 1);

    // Pads the right and bottom of [N, 3, H, W] to multiples of Unit: reflect where the frame allows it, else replicate.
    private Tensor<T> PadToUnit(Tensor<T> frames)
    {
        int h = frames.Shape[2], w = frames.Shape[3], unit = Unit;
        int padH = (unit - h % unit) % unit, padW = (unit - w % unit) % unit;
        if (padH == 0 && padW == 0) return frames;
        var mode = padH < h && padW < w ? PadMode.Reflect : PadMode.Replicate;
        return Engine.PadNd(frames, new[] { 0, padW, 0, padH }, mode, NumOps.Zero);
    }

    // Repeats the last of F frames [F, ...] until the count is a multiple of the clip length.
    private Tensor<T> FillClips(Tensor<T> frames)
    {
        int f = frames.Shape[0], n = _options.NumFrames, extra = (n - f % n) % n;
        if (extra == 0) return frames;
        var shape = frames.Shape.ToArray();
        shape[0] = extra;
        return Engine.TensorConcatenate(new[] { frames, Engine.TensorBroadcastTo(Engine.TensorNarrow(frames, 0, f - 1, 1), shape) }, 0);
    }

    // The posterior sample of frames [N, 3, H, W] in [−1, 1]: mean + σ·ε of quant_conv(encoder(x)), log-variance clamped
    // to [−30, 20] as DiagonalGaussianDistribution does; optionally the encoder's level outputs for the CFW modules.
    private Tensor<T> Encode(Tensor<T> frames, Random random, List<Tensor<T>>? features = null)
    {
        var encoder = _encoder ?? throw NotInitialized();
        var quant = _quantConv ?? throw NotInitialized();
        var moments = quant.Forward(encoder.Forward(frames, features));
        var mean = Engine.TensorNarrow(moments, 1, 0, Latent);
        var logVariance = Engine.TensorClamp(Engine.TensorNarrow(moments, 1, Latent, Latent), NumOps.FromDouble(-30.0), NumOps.FromDouble(20.0));
        var std = Engine.TensorExp(Engine.TensorMultiplyScalar(logVariance, NumOps.FromDouble(0.5)));
        return Engine.TensorAdd(mean, Engine.TensorMultiply(std, Gaussian(mean.Shape.ToArray(), random)));
    }

    private Tensor<T> Scaled(Tensor<T> latents, double factor) => Engine.TensorMultiplyScalar(latents, NumOps.FromDouble(factor));

    // q(x_t | x_0) at per-frame timesteps.
    private Tensor<T> Noised(Tensor<T> x0, Tensor<T> noise, IReadOnlyList<int> timesteps)
    {
        int n = x0.Shape[0], per = x0.Length / n;
        var keep = new Tensor<T>(x0.Shape.ToArray());
        var add = new Tensor<T>(x0.Shape.ToArray());
        for (int b = 0; b < n; b++)
        {
            double abar = _alphasCumprod[timesteps[b]];
            for (int i = 0; i < per; i++)
            {
                keep[b * per + i] = NumOps.FromDouble(Math.Sqrt(abar));
                add[b * per + i] = NumOps.FromDouble(Math.Sqrt(1.0 - abar));
            }
        }
        return Engine.TensorAdd(Engine.TensorMultiply(x0, keep), Engine.TensorMultiply(noise, add));
    }

    // ε for noisy latents of frames at per-frame timesteps, conditioned on the LR latents through the structure encoder.
    private Tensor<T> Epsilon(Tensor<T> noisy, IReadOnlyList<int> timesteps, Tensor<T> lowLatents)
    {
        var t = timesteps.Select(v => (double)v).ToArray();
        var condition = Structure.Forward(Engine, lowLatents, t);
        return UNet.Forward(Engine, noisy, t, Context(noisy.Shape[0]), condition, _options.NumFrames);
    }

    /// <summary>The flows and visibility masks between neighbouring frames of one clip, at the latent resolution.</summary>
    private sealed class ClipMotion
    {
        internal ClipMotion(List<Tensor<T>> toNext, List<Tensor<T>> toPrevious, List<Tensor<T>> visibleNext, List<Tensor<T>> visiblePrevious)
        {
            ToNext = toNext;
            ToPrevious = toPrevious;
            VisibleNext = visibleNext;
            VisiblePrevious = visiblePrevious;
        }

        internal List<Tensor<T>> ToNext { get; }

        internal List<Tensor<T>> ToPrevious { get; }

        internal List<Tensor<T>> VisibleNext { get; }

        internal List<Tensor<T>> VisiblePrevious { get; }
    }

    private Tensor<T> Visible(Tensor<T> occlusion) => Engine.TensorAddScalar(Engine.TensorNegate(occlusion), NumOps.One);

    // RAFT on the LQ clip brought back to a quarter of its size in [0, 1], flows resized to the latents (release script).
    private ClipMotion Motion(Tensor<T> lq, int latentHeight, int latentWidth)
    {
        int t = lq.Shape[0], h = lq.Shape[2], w = lq.Shape[3];
        var small = Engine.Interpolate(ToUnit(lq), new[] { Math.Max(1, h / 4), Math.Max(1, w / 4) }, InterpolateMode.Bicubic, false);
        var toNext = new List<Tensor<T>>();
        var toPrevious = new List<Tensor<T>>();
        var visibleNext = new List<Tensor<T>>();
        var visiblePrevious = new List<Tensor<T>>();
        for (int i = 0; i < t - 1; i++)
        {
            var current = Engine.TensorNarrow(small, 0, i, 1);
            var next = Engine.TensorNarrow(small, 0, i + 1, 1);
            var forward = MgldFlow.Resize(Engine, Raft.Flow(Engine, current, next, _options.FlowIterations), latentHeight, latentWidth);
            var backward = MgldFlow.Resize(Engine, Raft.Flow(Engine, next, current, _options.FlowIterations), latentHeight, latentWidth);
            var (forwardOcclusion, backwardOcclusion) = MgldFlow.Occlusion(Engine, forward, backward);
            toNext.Add(forward);
            toPrevious.Add(backward);
            visibleNext.Add(Visible(forwardOcclusion));
            visiblePrevious.Add(Visible(backwardOcclusion));
        }
        return new ClipMotion(toNext, toPrevious, visibleNext, visiblePrevious);
    }

    // −g·log σ²_t·∇L on the latents of one clip, L the masked warping error between neighbouring latents.
    private Tensor<T> Guide(Tensor<T> latents, ClipMotion motion, double logVariance)
    {
        if (latents.Shape[0] < 2 || _options.GuidanceScale == 0) return latents;
        var gradient = MgldFlow.WarpingLossGradient(latents, motion.ToNext, motion.ToPrevious, motion.VisibleNext, motion.VisiblePrevious);
        return Engine.TensorSubtract(latents, Engine.TensorMultiplyScalar(gradient, NumOps.FromDouble(_options.GuidanceScale * logVariance)));
    }

    /// <summary>
    /// DDPM sampling of one clip's latents (the release's <c>sample_canvas</c> on a single tile): from the LR latent
    /// noised to the last timestep, each respaced step predicts ε at its original timestep, takes the posterior mean
    /// (plus noise but at the last step) and applies the motion guidance.
    /// </summary>
    private Tensor<T> Sample(Tensor<T> lowLatents, ClipMotion motion, Random random)
    {
        int n = lowLatents.Shape[0], last = _options.NumTrainTimesteps - 1;
        var shape = lowLatents.Shape.ToArray();
        var timesteps = SpacedTimesteps(_options.NumTrainTimesteps, _options.SamplingSteps);
        int s = timesteps.Length;
        var abar = timesteps.Select(t => _alphasCumprod[t]).ToArray();
        var betas = new double[s];
        var variance = new double[s];
        for (int k = 0; k < s; k++)
        {
            double prev = k == 0 ? 1.0 : abar[k - 1];
            betas[k] = 1.0 - abar[k] / prev;
            variance[k] = betas[k] * (1.0 - prev) / (1.0 - abar[k]);
        }
        // The x_T the release starts from: the LR latent noised with the original schedule at the last timestep.
        var x = Noised(lowLatents, Gaussian(shape, random), Enumerable.Repeat(last, n).ToArray());
        for (int k = s - 1; k >= 0; k--)
        {
            // posterior_log_variance_clipped: the first step borrows the second's variance, which is never zero.
            double logVariance = Math.Log(k == 0 && s > 1 ? variance[1] : variance[k]);
            using (new NoGradScope<T>())
            {
                double prev = k == 0 ? 1.0 : abar[k - 1];
                var eps = Epsilon(x, Enumerable.Repeat(timesteps[k], n).ToArray(), lowLatents);
                var x0 = Engine.TensorSubtract(Scaled(x, Math.Sqrt(1.0 / abar[k])), Scaled(eps, Math.Sqrt(1.0 / abar[k] - 1.0)));
                double coef1 = betas[k] * Math.Sqrt(prev) / (1.0 - abar[k]);
                double coef2 = (1.0 - prev) * Math.Sqrt(1.0 - betas[k]) / (1.0 - abar[k]);
                x = Engine.TensorAdd(Scaled(x0, coef1), Scaled(x, coef2));
                if (k > 0) x = Engine.TensorAdd(x, Scaled(Gaussian(shape, random), Math.Exp(0.5 * logVariance)));
            }
            // A single sampling step has a zero posterior variance and so no finite guidance step.
            if (!double.IsInfinity(logVariance) && !double.IsNaN(logVariance)) x = Guide(x, motion, logVariance);
        }
        return x;
    }

    // AdaIN per frame and channel against the LQ frames (unbiased = False, ε 1e-5 on the variance).
    private Tensor<T> ColorFix(Tensor<T> frames, Tensor<T> lq)
    {
        var shape = frames.Shape.ToArray();
        var axes = new[] { 2, 3 };
        Tensor<T> Mean(Tensor<T> x) => Engine.TensorBroadcastTo(Engine.ReduceMean(x, axes, true), shape);
        Tensor<T> Std(Tensor<T> x, Tensor<T> mean)
            => Engine.TensorSqrt(Engine.TensorAddScalar(Mean(Engine.TensorSquare(Engine.TensorSubtract(x, mean))), NumOps.FromDouble(1e-5)));
        var mean = Mean(frames);
        var styleMean = Mean(lq);
        var normalized = Engine.TensorDivide(Engine.TensorSubtract(frames, mean), Std(frames, mean));
        return Engine.TensorAdd(Engine.TensorMultiply(normalized, Std(lq, styleMean)), styleMean);
    }

    // One clip of padded LQ frames [N, 3, H, W] in [−1, 1] to frames in [−1, 1].
    private Tensor<T> RunClip(Tensor<T> lq, Random random)
    {
        var features = new List<Tensor<T>>();
        Tensor<T> low;
        ClipMotion motion;
        using (new NoGradScope<T>())
        {
            low = Scaled(Encode(lq, random, features), _options.LatentScale);
            motion = Motion(lq, low.Shape[2], low.Shape[3]);
        }
        var latents = Sample(low, motion, random);
        using (new NoGradScope<T>())
        {
            var frames = Decoder.Forward(Engine, Scaled(latents, 1.0 / _options.LatentScale), features, _options.FusionWeight, _options.NumFrames);
            if (_options.ColorFix) frames = ColorFix(frames, lq);
            return frames;
        }
    }

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the output size, in [0, 1].
    private Tensor<T> RunVideo(Tensor<T> frames, Random random)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        double upscale = Math.Max(_options.MinimumSide > 0 ? _options.MinimumSide / (double)Math.Min(h, w) : 0.0, s);
        int uh = (int)(h * upscale), uw = (int)(w * upscale);
        var outputs = new List<Tensor<T>>();
        for (int i = 0; i < b; i++)
        {
            var video = FillClips(Engine.Reshape(Engine.TensorNarrow(clips, 0, i, 1), new[] { f, 3, h, w }));
            var lq = Engine.TensorClamp(ToSigned(Engine.Interpolate(video, new[] { uh, uw }, InterpolateMode.Bicubic, false)), NumOps.FromDouble(-1.0), NumOps.One);
            var padded = PadToUnit(lq);
            var results = new List<Tensor<T>>();
            for (int start = 0; start < video.Shape[0]; start += _options.NumFrames)
                results.Add(RunClip(Engine.TensorNarrow(padded, 0, start, _options.NumFrames), random));
            var sr = ToUnit(Engine.TensorSlice(Engine.TensorConcatenate(results.ToArray(), 0), new[] { 0, 0, 0, 0 }, new[] { f, 3, uh, uw }));
            if (uh != h * s || uw != w * s)
                sr = Engine.TensorClamp(Engine.Interpolate(sr, new[] { h * s, w * s }, InterpolateMode.Bicubic, false), NumOps.Zero, NumOps.One);
            outputs.Add(Engine.Reshape(sr, new[] { 1, f, 3, h * s, w * s }));
        }
        var output = Engine.TensorConcatenate(outputs.ToArray(), 0);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
        => _usingCustomLayers ? base.Forward(input) : RunVideo(input, InferenceRandom());

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunVideo(PreprocessFrames(input), InferenceRandom());

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> PaperAdamW()
        => new(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
        });

    private AdamOptimizer<T, Tensor<T>, Tensor<T>> PaperAdam()
        => new(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.DecoderLearningRate,
            Beta1 = 0.5,
            Beta2 = 0.9,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiffusionOptimizer()
        => _diffusionOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "diffusion");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DecoderOptimizer()
        => _decoderOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdam(), "decoder");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdam(), "discriminator");

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), false);

    /// <summary>One training batch prepared as the release's <c>get_input</c> does.</summary>
    private sealed class Batch
    {
        internal Batch(Tensor<T> lq, Tensor<T> high, Tensor<T> lowLatents, Tensor<T> highLatents)
        {
            Lq = lq;
            High = high;
            LowLatents = lowLatents;
            HighLatents = highLatents;
        }

        /// <summary>The LQ frames upsampled to the ground truth's size, [N, 3, H, W] in [−1, 1].</summary>
        internal Tensor<T> Lq { get; }

        /// <summary>The ground-truth frames [N, 3, H, W] in [−1, 1].</summary>
        internal Tensor<T> High { get; }

        /// <summary>The LQ frames' scaled latents.</summary>
        internal Tensor<T> LowLatents { get; }

        /// <summary>The ground truth's scaled latents.</summary>
        internal Tensor<T> HighLatents { get; }
    }

    private Batch Prepare(Tensor<T> input, Tensor<T> target, Random random)
    {
        var low = PreprocessFrames(input);
        var high = NormalizeFrames(target);
        if (low.Rank == 5)
        {
            low = Engine.Reshape(low, new[] { low.Shape[0] * low.Shape[1] }.Concat(low._shape.Skip(2)).ToArray());
            high = Engine.Reshape(high, new[] { high.Shape[0] * high.Shape[1] }.Concat(high._shape.Skip(2)).ToArray());
        }
        using var noGrad = new NoGradScope<T>();
        low = FillClips(low);
        high = FillClips(high);
        int h = high.Shape[2], w = high.Shape[3];
        var lq = Engine.TensorClamp(ToSigned(Engine.Interpolate(low, new[] { h, w }, InterpolateMode.Bicubic, false)), NumOps.FromDouble(-1.0), NumOps.One);
        lq = PadToUnit(lq);
        var truth = PadToUnit(Engine.TensorClamp(ToSigned(high), NumOps.FromDouble(-1.0), NumOps.One));
        return new Batch(lq, truth, Scaled(Encode(lq, random), _options.LatentScale), Scaled(Encode(truth, random), _options.LatentScale));
    }

    // The diffusion objective: ε-prediction MSE at per-frame uniform timesteps (p_losses with the eps parameterization).
    private Tensor<T> DiffusionLoss(Batch batch, Random random)
    {
        int n = batch.HighLatents.Shape[0];
        var timesteps = Enumerable.Range(0, n).Select(_ => random.Next(_options.NumTrainTimesteps)).ToArray();
        return DiffusionLoss(batch, timesteps, Gaussian(batch.HighLatents.Shape.ToArray(), random));
    }

    private Tensor<T> DiffusionLoss(Batch batch, int[] timesteps, Tensor<T> noise)
    {
        var noisy = Noised(batch.HighLatents, noise, timesteps);
        var diff = Engine.TensorSubtract(Epsilon(noisy, timesteps, batch.LowLatents), noise);
        return Mean(Engine.TensorSquare(diff));
    }

    /// <summary>The number of timesteps on the even grid the declared diffusion objective averages over.</summary>
    private const int ObjectiveTimesteps = 8;

    // The ε-objective is an expectation over the timestep and the noise; one draw of it moves by far more between calls
    // than a few optimizer steps move the model. The declared objective is the mean over an even grid of timesteps
    // with seeded noise, so two evaluations of an unchanged model agree.
    private T DiffusionObjective(Batch batch, Random random)
    {
        int n = batch.HighLatents.Shape[0];
        double sum = 0;
        for (int q = 0; q < ObjectiveTimesteps; q++)
        {
            int step = (int)((q + 0.5) * _options.NumTrainTimesteps / ObjectiveTimesteps);
            var timesteps = Enumerable.Repeat(step, n).ToArray();
            sum += NumOps.ToDouble(DiffusionLoss(batch, timesteps, Gaussian(batch.HighLatents.Shape.ToArray(), random))[0]);
        }
        return NumOps.FromDouble(sum / ObjectiveTimesteps);
    }

    /// <summary>The decoder stage's inputs: the diffusion's sampled latents and the LQ encoder's features.</summary>
    private sealed class DecoderBatch
    {
        internal DecoderBatch(Batch batch, Tensor<T> latents, IReadOnlyList<Tensor<T>> features, Tensor<T>? edgeWeights,
            IReadOnlyList<Tensor<T>> toNext, IReadOnlyList<Tensor<T>> toPrevious, IReadOnlyList<Tensor<T>> visibleNext,
            IReadOnlyList<Tensor<T>> visiblePrevious)
        {
            Source = batch;
            Latents = latents;
            Features = features;
            EdgeWeights = edgeWeights;
            ToNext = toNext;
            ToPrevious = toPrevious;
            VisibleNext = visibleNext;
            VisiblePrevious = visiblePrevious;
        }

        internal Batch Source { get; }

        internal Tensor<T> Latents { get; }

        internal IReadOnlyList<Tensor<T>> Features { get; }

        internal Tensor<T>? EdgeWeights { get; }

        internal IReadOnlyList<Tensor<T>> ToNext { get; }

        internal IReadOnlyList<Tensor<T>> ToPrevious { get; }

        internal IReadOnlyList<Tensor<T>> VisibleNext { get; }

        internal IReadOnlyList<Tensor<T>> VisiblePrevious { get; }
    }

    // The release decodes latents the trained diffusion produced from the LQ clips; they are sampled here the same way.
    // The warping loss's flows and edge weights come from the ground truth (SPyNet, Sobel), per clip.
    private DecoderBatch PrepareDecoder(Batch batch, Random random)
    {
        int n = batch.Lq.Shape[0], clip = _options.NumFrames;
        var latents = new List<Tensor<T>>();
        var toNext = new List<Tensor<T>>();
        var toPrevious = new List<Tensor<T>>();
        var visibleNext = new List<Tensor<T>>();
        var visiblePrevious = new List<Tensor<T>>();
        var features = new List<Tensor<T>>();
        using (new NoGradScope<T>()) Encode(batch.Lq, random, features);
        for (int start = 0; start < n; start += clip)
        {
            var lq = Engine.TensorNarrow(batch.Lq, 0, start, clip);
            var low = Engine.TensorNarrow(batch.LowLatents, 0, start, clip);
            ClipMotion motion;
            using (new NoGradScope<T>()) motion = Motion(lq, low.Shape[2], low.Shape[3]);
            latents.Add(Scaled(Sample(low, motion, random), 1.0 / _options.LatentScale));
            if (_spyNet is null) continue;
            using (new NoGradScope<T>())
            {
                var truth = ToUnit(Engine.TensorNarrow(batch.High, 0, start, clip));
                for (int i = 0; i < clip - 1; i++)
                {
                    var current = Engine.TensorNarrow(truth, 0, i, 1);
                    var next = Engine.TensorNarrow(truth, 0, i + 1, 1);
                    var forward = _spyNet.Flow(Engine, Layers, current, next);
                    var backward = _spyNet.Flow(Engine, Layers, next, current);
                    var (forwardOcclusion, backwardOcclusion) = MgldFlow.Occlusion(Engine, forward, backward);
                    toNext.Add(forward);
                    toPrevious.Add(backward);
                    visibleNext.Add(Visible(forwardOcclusion));
                    visiblePrevious.Add(Visible(backwardOcclusion));
                }
            }
        }
        Tensor<T>? edges = null;
        if (_spyNet is not null)
            using (new NoGradScope<T>())
                edges = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(MgldFlow.Sobel(Engine, batch.High), NumOps.FromDouble(_options.EdgeWeight)), NumOps.One);
        return new DecoderBatch(batch, Engine.TensorConcatenate(latents.ToArray(), 0), features, edges, toNext, toPrevious, visibleNext, visiblePrevious);
    }

    private Tensor<T> Reconstruct(DecoderBatch batch)
        => Decoder.Forward(Engine, batch.Latents, batch.Features, _options.TrainingFusionWeight, _options.NumFrames);

    // The reconstruction term: mean(|x − x̂| + λ_p·LPIPS) divided by the frame count, as LPIPSWithDiscriminator does.
    private Tensor<T> ReconstructionLoss(Tensor<T> reconstruction, Tensor<T> truth)
    {
        var loss = Mean(Engine.TensorAbs(Engine.TensorSubtract(truth, reconstruction)));
        if (_lpips is not null)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(_lpips.Loss(Engine, Layers, reconstruction, truth), NumOps.FromDouble(_options.PerceptualWeight)));
        return Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(1.0 / truth.Shape[0]));
    }

    // L1 between the ground truth's and the reconstruction's frame differences within each clip, mean over the
    // differences and divided by their count.
    private Tensor<T> DifferenceLoss(Tensor<T> reconstruction, Tensor<T> truth)
    {
        int n = truth.Shape[0], clip = _options.NumFrames;
        if (clip < 2) return new Tensor<T>(new[] { 1 });
        var diffs = new List<Tensor<T>>();
        for (int start = 0; start < n; start += clip)
            for (int i = 0; i < clip - 1; i++)
            {
                Tensor<T> Delta(Tensor<T> x) => Engine.TensorSubtract(Engine.TensorNarrow(x, 0, start + i, 1), Engine.TensorNarrow(x, 0, start + i + 1, 1));
                diffs.Add(Engine.TensorSubtract(Delta(truth), Delta(reconstruction)));
            }
        var all = Engine.TensorConcatenate(diffs.ToArray(), 0);
        return Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(all)), NumOps.FromDouble(1.0 / all.Shape[0]));
    }

    private Tensor<T> TemporalLoss(DecoderBatch batch, Tensor<T> reconstruction)
    {
        if (batch.EdgeWeights is null) return new Tensor<T>(new[] { 1 });
        int n = reconstruction.Shape[0], clip = _options.NumFrames, pairs = clip - 1;
        Tensor<T>? loss = null;
        for (int c = 0, start = 0; start < n; start += clip, c++)
        {
            var range = Enumerable.Range(c * pairs, pairs);
            var part = MgldFlow.WarpingLoss(Engine, Engine.TensorNarrow(reconstruction, 0, start, clip),
                range.Select(i => batch.ToNext[i]).ToList(), range.Select(i => batch.ToPrevious[i]).ToList(),
                range.Select(i => batch.VisibleNext[i]).ToList(), range.Select(i => batch.VisiblePrevious[i]).ToList(),
                Engine.TensorNarrow(batch.EdgeWeights, 0, start, clip));
            loss = loss is null ? part : Engine.TensorAdd(loss, part);
        }
        return loss ?? new Tensor<T>(new[] { 1 });
    }

    // L1 + LPIPS, the difference and the warping terms, without the adversarial term.
    private Tensor<T> DecoderLoss(DecoderBatch batch, Tensor<T> reconstruction)
    {
        var truth = batch.Source.High;
        var loss = ReconstructionLoss(reconstruction, truth);
        loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(DifferenceLoss(reconstruction, truth), NumOps.FromDouble(_options.DifferenceWeight)));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(TemporalLoss(batch, reconstruction), NumOps.FromDouble(_options.TemporalWeight)));
    }

    // taming's calculate_adaptive_weight: ‖∇_W nll‖ / (‖∇_W g‖ + 1e-4) at the decoder's last layer, clamped to [0, 1e4],
    // times the discriminator weight; computed before the step and held constant, as its detach makes it.
    private double AdaptiveWeight(DecoderBatch batch)
    {
        var last = Decoder.LastLayer;
        double Norm(Func<Tensor<T>, Tensor<T>> loss)
        {
            using var tape = new GradientTape<T>();
            var value = loss(Reconstruct(batch));
            var grads = tape.ComputeGradients(value, new[] { last });
            if (!grads.TryGetValue(last, out var g)) return 0.0;
            double sum = 0;
            for (int i = 0; i < g.Length; i++) sum += NumOps.ToDouble(g[i]) * NumOps.ToDouble(g[i]);
            return Math.Sqrt(sum);
        }
        double nll = Norm(r => ReconstructionLoss(r, batch.Source.High));
        double adversarial = Norm(r => Engine.TensorNegate(Mean(Discriminator.Forward(r))));
        return Math.Min(Math.Max(nll / (adversarial + 1e-4), 0.0), 1e4) * _options.DiscriminatorWeight;
    }

    // The hinge loss: 0.5·(mean ReLU(1 − D(real)) + mean ReLU(1 + D(fake))).
    private Tensor<T> DiscriminatorLoss(Tensor<T> truth, Tensor<T> fake)
    {
        var real = Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(Discriminator.Forward(truth)), NumOps.One)));
        var generated = Mean(Engine.ReLU(Engine.TensorAddScalar(Discriminator.Forward(fake), NumOps.One)));
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(real, generated), NumOps.FromDouble(0.5));
    }

    /// <inheritdoc />
    /// <remarks>
    /// The release's two runs in turn: the diffusion (SPADE, the temporal modules and the structure encoder, ε-prediction)
    /// for <see cref="MGLDVSROptions.DiffusionIterations"/> steps, then the video decoder and, from
    /// <see cref="MGLDVSROptions.DiscriminatorStart"/> decoder steps on, the discriminator.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        int stage = TrainingStage, decoderStep = _trainingSteps - _options.DiffusionIterations;
        _trainingSteps++;
        var random = RandomHelper.CreateSeededRandom(_random.Next());
        var batch = Prepare(input, expectedOutput, random);
        var sources = new[] { input, expectedOutput };
        if (stage == 0)
            return new[]
            {
                new TrainingObjective<T>(sources, _ => DiffusionLoss(batch, random)) { Optimizer = DiffusionOptimizer(), Parameters = () => DiffusionParameters },
            };

        var decoderBatch = PrepareDecoder(batch, random);
        bool adversarial = decoderStep >= _options.DiscriminatorStart && _options.DiscriminatorWeight > 0;
        double weight = adversarial ? AdaptiveWeight(decoderBatch) : 0.0;
        var objectives = new List<TrainingObjective<T>>
        {
            new(sources, _ =>
            {
                var reconstruction = Reconstruct(decoderBatch);
                var loss = DecoderLoss(decoderBatch, reconstruction);
                if (!adversarial) return loss;
                var g = Engine.TensorNegate(Mean(Discriminator.Forward(reconstruction)));
                return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(g, NumOps.FromDouble(weight)));
            }) { Optimizer = DecoderOptimizer(), Parameters = () => DecoderParameters },
        };
        if (adversarial)
        {
            Tensor<T> fake;
            using (new NoGradScope<T>()) fake = Reconstruct(decoderBatch);
            objectives.Add(new TrainingObjective<T>(sources, _ => DiscriminatorLoss(decoderBatch.Source.High, fake))
            {
                Optimizer = DiscriminatorOptimizer(),
                Parameters = () => DiscriminatorParameters,
            });
        }
        return objectives;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.DiffusionDenoising;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The current stage's loss with every random draw from a fixed seed: the ε-objective averaged over an even grid of timesteps, the decoder's without the adversarial term.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (_usingCustomLayers || !_useNativeMode)
            using (new NoGradScope<T>())
                return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var random = RandomHelper.CreateSeededRandom(1729);
        var batch = Prepare(input, target, random);
        if (TrainingStage == 0)
            using (new NoGradScope<T>())
                return DiffusionObjective(batch, random);
        var decoderBatch = PrepareDecoder(batch, random);
        using (new NoGradScope<T>())
            return DecoderLoss(decoderBatch, Reconstruct(decoderBatch))[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? DiffusionOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    // Each step draws its batch on the host (the VAE posterior sample, the timesteps and the noise; in the decoder
    // stage the sampled latents and the flows), which a compiled plan would freeze at its first step.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

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
            Name = _useNativeMode ? "MGLDVSR-Native" : "MGLDVSR-ONNX",
            Description = "MGLD-VSR motion-guided latent diffusion video super-resolution on SD 2.1 (Yang et al. 2024)",
            Complexity = _options.UNetMultipliers.Length
        };
        m.AdditionalInfo["UNetChannels"] = _options.UNetChannels.ToString();
        m.AdditionalInfo["SamplingSteps"] = _options.SamplingSteps.ToString();
        m.AdditionalInfo["GuidanceScale"] = _options.GuidanceScale.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MGLDVSR<T>));
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
