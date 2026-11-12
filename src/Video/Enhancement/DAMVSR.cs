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
/// DAM-VSR: video super-resolution on Stable Video Diffusion that takes appearance from an image super-resolution model
/// and motion from the low-resolution video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is Kong et al.'s DAM-VSR (SIGGRAPH 2025, arXiv 2507.01012) on Stable Video Diffusion as diffusers releases
/// it. An image super-resolution model upscales the clip's first frame; its VAE latent, repeated over the frames, is
/// concatenated with the noisy latent, and its CLIP ViT-H/14 embedding is the U-Net's cross-attention context. A video
/// ControlNet, a copy of the U-Net's encoder with its temporal layers, sees the same input plus the low-resolution video
/// through ControlNet's conditioning embedding, and its zero-initialized outputs are added to the U-Net's skips and middle
/// block. The denoiser is SVD's <c>UNetSpatioTemporalConditionModel</c> with EDM v-prediction preconditioning.
/// </para>
/// <para>
/// Sampling enters a 30-step Euler schedule 60% of the way down from the noised latent of the bicubically upscaled clip
/// (SDEdit). With motion-aligned bidirectional sampling each step also runs the clip backwards from its super-resolved last
/// frame, reusing the forward pass's temporal attention maps rotated 180 degrees with the backward pass's own temporal value
/// and output projections, and averages the two predictions. Training is the paper's three stages: the ControlNet, then
/// the backward projections, then the VAE decoder on L2, perceptual and GAN losses of 8-step samples. The paper releases
/// no code. Where it is silent this follows SVD's pipeline: added time ids (fps - 1, motion bucket 127, noise
/// augmentation 0.02), unscaled reference latents, EDM training noise log σ ~ N(0.7, 1.6²), the ground-truth first frame as
/// the training reference, no classifier-free guidance (the low-resolution video conditions every frame), and
/// RealBasicVSR's discriminator. The decoder is fine-tuned directly rather than through LoRA adapters, and the image
/// super-resolution model is supplied by the caller (<see cref="ReferenceUpscaler"/>), bicubic when none is.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAM-VSR splits video upscaling in two. A strong photo upscaler makes one frame sharp and
/// detailed, the low-resolution video says how everything moves, and a video diffusion model generates the remaining
/// frames so they match the sharp frame and follow the motion.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 14, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new DAMVSR&lt;float&gt;(arch, new DAMVSROptions());
/// model.ReferenceUpscaler = frame =&gt; myImageSuperResolver.Upscale(frame);
/// var upscaled = model.Upscale(lowResClip);      // [14, 3, 64, 64] -> [14, 3, 256, 256], pixel values 0-255
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DAM-VSR: Disentanglement of Appearance and Motion for Video Super-Resolution",
    "https://arxiv.org/abs/2507.01012",
    Year = 2025,
    Authors = "Zhe Kong, Le Li, Yong Zhang, Feng Gao, Shaoshu Yang, Tao Wang, Kaihao Zhang, Zhuoliang Kang, Xiaoming Wei, Guanying Chen, Wenhan Luo")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 8e-5, WeightDecay = 0.01, Component = "controlnet",
                Source = "Kong et al. 2025, Sec. 4.1: AdamW, batch 8, a constant 8e-5; 60k iterations train the ControlNet.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 8e-5, WeightDecay = 0.01, Component = "backward",
                Source = "Kong et al. 2025, Sec. 4.1: 45k iterations train the backward pass's temporal W_v and W_o.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 8e-5, WeightDecay = 0.01, Component = "decoder",
                Source = "Kong et al. 2025, Sec. 4.1: 10k iterations train the VAE decoder.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 8e-5, WeightDecay = 0.01, Component = "discriminator",
                Source = "Kong et al. 2025, Sec. 4.1: the decoder stage's GAN loss; the discriminator's settings are unstated.")]
public partial class DAMVSR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly DAMVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _controlOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _backwardOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _decoderOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private SvdUNet<T>? _unet;
    private DamVsrControlNet<T>? _controlNet;
    private SvdImageEncoder<T>? _imageEncoder;
    // The backward pass's temporal value and output projections, one pair per temporal block of the U-Net.
    private (int Value, int Output)[] _backwardTemporal = Array.Empty<(int, int)>();
    private int _unetFirst, _unetEnd, _controlFirst, _controlEnd, _backwardFirst, _backwardEnd;
    private KlEncoder<T>? _vaeEncoder;
    private KlConv2d<T>? _quantConv;
    private KlTemporalDecoder<T>? _vaeDecoder;
    private SpectralNormUNetDiscriminator<T>? _discriminator;
    private Vgg19PerceptualBlock<T>? _vgg;
    private Random _random;
    private Tensor<T>? _lastFake;

    /// <summary>
    /// The image super-resolution model that makes the reference frames: [3, h, w] in [0, 1] to [3, s·h, s·w] in [0, 1].
    /// The paper uses ResShift for synthetic and SUPIR or InvSR for real-world video; bicubic upscaling when null.
    /// </summary>
    public Func<Tensor<T>, Tensor<T>>? ReferenceUpscaler { get; set; }

    /// <summary>Creates a DAM-VSR model in ONNX inference mode.</summary>
    public DAMVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, DAMVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new DAMVSROptions() : new DAMVSROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DAM-VSR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's and SVD's defaults when null.</param>
    /// <param name="optimizer">The ControlNet stage's optimizer; the paper's AdamW when null. The later stages always use
    /// AdamW at <see cref="DAMVSROptions.LearningRate"/>.</param>
    public DAMVSR(NeuralNetworkArchitecture<T> architecture, DAMVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _controlOptimizer = optimizer;
        _options = options is null ? new DAMVSROptions() : new DAMVSROptions(options);
        Options = _options;
        var o = _options;
        if (o.UNetChannels is not { Length: >= 2 } || o.UNetHeads is null || o.UNetHeads.Length != o.UNetChannels.Length
            || o.UNetChannels.Any(c => c < 32 || c % 32 != 0) || o.UNetChannels.Zip(o.UNetHeads, (c, h) => h < 1 || c % h != 0).Any(bad => bad))
            throw new ArgumentOutOfRangeException(nameof(options), "UNetChannels need two or more multiples of 32, each divisible by its UNetHeads entry.");
        if (o.VaeChannels is not { Length: >= 1 } || o.VaeChannels.Any(c => c < 32 || c % 32 != 0))
            throw new ArgumentOutOfRangeException(nameof(options), "VaeChannels must be multiples of 32.");
        if (o.ScaleFactor < 1 || o.NumInferenceSteps < 1 || o.SdeditStrength <= 0 || o.SdeditStrength > 1 || o.DecoderSamplingSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor, step counts and SdeditStrength (0, 1] must be valid.");
        if (o.ImageEncoderSize % o.ImageEncoderPatchSize != 0 || o.ImageEncoderWidth % o.ImageEncoderHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The image encoder's size must divide into patches and its width into heads.");
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
        state.DeclareDoubleVector(nameof(DAMVSR<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
            restored => _trainingSteps = restored is { Length: > 0 } ? (int)restored[0] : 0);
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent("DAMVSR.VaeEncoder", _vaeEncoder);
        RegisterParameterComponent("DAMVSR.VaeQuantConv", _quantConv);
        RegisterParameterComponent("DAMVSR.VaeDecoder", _vaeDecoder);
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
        _unetFirst = Layers.Count;
        _unet = new SvdUNet<T>(Layers, 4, o.UNetChannels, o.UNetHeads, o.LayersPerBlock, o.TransformerDepth, o.AddedTimeEmbedDim);
        _unetEnd = Layers.Count;
        _backwardFirst = Layers.Count;
        var backward = new List<(int, int)>();
        foreach (var transformer in _unet.Transformers())
            foreach (var _ in transformer.TemporalBlocks)
            {
                int value = Layers.Count;
                Layers.Add(RvrtBlocks.Linear<T>(transformer.Channels, bias: false));
                int output = Layers.Count;
                Layers.Add(RvrtBlocks.Linear<T>(transformer.Channels));
                backward.Add((value, output));
            }
        _backwardTemporal = backward.ToArray();
        _backwardEnd = Layers.Count;
        _controlFirst = Layers.Count;
        _controlNet = new DamVsrControlNet<T>(Layers, o.UNetChannels, o.UNetHeads, o.LayersPerBlock, o.TransformerDepth, o.AddedTimeEmbedDim,
            Log2(1 << (o.VaeChannels.Length - 1)));
        _controlEnd = Layers.Count;
        _imageEncoder = new SvdImageEncoder<T>(Layers, o.ImageEncoderSize, o.ImageEncoderPatchSize, o.ImageEncoderWidth, o.ImageEncoderDepth,
            o.ImageEncoderHeads, o.ImageEncoderMlpDim, o.ImageEmbedDim);
        _vaeEncoder = new KlEncoder<T>(3, 4, o.VaeChannels, o.VaeLayersPerBlock, 32);
        _quantConv = new KlConv2d<T>(8, 8, 1);
        _vaeDecoder = new KlTemporalDecoder<T>(4, 3, o.VaeChannels, o.VaeLayersPerBlock, 32);
        _discriminator = new SpectralNormUNetDiscriminator<T>(Layers, o.DiscriminatorChannels, _random);
        // A zero perceptual weight leaves the 20M-parameter VGG19 out entirely rather than carrying it unused.
        if (o.PerceptualWeight > 0) _vgg = new Vgg19PerceptualBlock<T>(Layers);
    }

    private static int Log2(int value)
    {
        int log = 0;
        while ((1 << log) < value) log++;
        return log;
    }

    private int LatentFactor => 1 << (_options.VaeChannels.Length - 1);

    // The smallest output side the U-Net's down stages and the VAE both divide evenly.
    private int MinimumSide => LatentFactor << (_options.UNetChannels.Length - 1);

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        // The discriminator halves the frames three times, so the side must also divide by 8 (both are powers of two).
        int unit = Math.Max(MinimumSide, 8);
        int side = (Math.Max(unit, 2 * _options.ScaleFactor) + unit - 1) / unit * unit;
        int low = Math.Max(1, side / _options.ScaleFactor);
        var output = RunClips(new Tensor<T>(new[] { 1, 2, 3, low, low }), 1, InferenceRandom());
        var frames = Engine.Reshape(output, new[] { 2, 3, output.Shape[3], output.Shape[4] });
        Discriminator.Run(Engine, Layers, frames, training: false);
        _vgg?.Loss(Engine, Layers, frames, frames);
        MarkLayerShapesResolved();
    }

    private SvdUNet<T> UNet => _unet ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private DamVsrControlNet<T> ControlNet => _controlNet ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private SvdImageEncoder<T> ImageEncoder => _imageEncoder ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private SpectralNormUNetDiscriminator<T> Discriminator
        => _discriminator ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private KlEncoder<T> VaeEncoder => _vaeEncoder ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private KlConv2d<T> QuantConv => _quantConv ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private KlTemporalDecoder<T> VaeDecoder => _vaeDecoder ?? throw new InvalidOperationException("DAM-VSR's native layers are not initialized.");

    private IReadOnlyList<Tensor<T>> ParametersOf(int first, int end)
        => TapeTrainingStep<T>.CollectParameters(Layers.Skip(first).Take(end - first));

    /// <summary>The ControlNet's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> ControlNetParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : ParametersOf(_controlFirst, _controlEnd);

    /// <summary>The U-Net's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> UNetParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : ParametersOf(_unetFirst, _unetEnd);

    /// <summary>The backward pass's temporal value and output projections.</summary>
    internal IReadOnlyList<Tensor<T>> BackwardParameters => _usingCustomLayers ? Array.Empty<Tensor<T>>() : ParametersOf(_backwardFirst, _backwardEnd);

    /// <summary>The VAE decoder's weights.</summary>
    internal IReadOnlyList<Tensor<T>> DecoderParameters
        => _vaeDecoder is null ? Array.Empty<Tensor<T>>() : _vaeDecoder.NamedTensors("").Select(named => named.Tensor).ToList();

    /// <summary>The discriminator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> DiscriminatorParameters
        => _discriminator is null ? Array.Empty<Tensor<T>>() : _discriminator.Parameters(Layers);

    /// <summary>The training stage the next step runs: 0 ControlNet, 1 backward projections, 2 VAE decoder.</summary>
    internal int TrainingStage => _trainingSteps < _options.ControlNetIterations ? 0
        : _trainingSteps < _options.ControlNetIterations + _options.BackwardIterations ? 1 : 2;

    #endregion

    #region Diffusion

    // One pass's conditioning: the CLIP context and the reference latent per frame, the low-resolution frames, the clip length.
    private sealed class PassCondition
    {
        internal PassCondition(Tensor<T> context, Tensor<T> reference, Tensor<T> lowRes, int frames)
        {
            Context = context;
            Reference = reference;
            LowRes = lowRes;
            Frames = frames;
        }

        internal Tensor<T> Context { get; }
        internal Tensor<T> Reference { get; }
        internal Tensor<T> LowRes { get; }
        internal int Frames { get; }
    }

    /// <summary>The Karras schedule of <paramref name="steps"/> σ from SigmaMax to SigmaMin, then 0 (EulerDiscreteScheduler).</summary>
    internal double[] Sigmas(int steps)
    {
        double max = Math.Pow(_options.SigmaMax, 1.0 / _options.KarrasRho), min = Math.Pow(_options.SigmaMin, 1.0 / _options.KarrasRho);
        var sigmas = new double[steps + 1];
        for (int i = 0; i < steps; i++)
            sigmas[i] = Math.Pow(steps == 1 ? max : max + i / (double)(steps - 1) * (min - max), _options.KarrasRho);
        return sigmas;
    }

    /// <summary>The schedule index SDEdit enters at: the last SdeditStrength of the <paramref name="steps"/> are run.</summary>
    internal int SdeditStart(int steps) => Math.Min(steps - 1, steps - (int)Math.Round(steps * _options.SdeditStrength));

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

    // Images [N, 3, H, W] in [-1, 1] to the posterior mode, unscaled: [N, 4, H / f, W / f].
    private Tensor<T> EncodeLatent(Tensor<T> images)
    {
        var moments = QuantConv.Forward(VaeEncoder.Forward(images));
        return Engine.TensorNarrow(moments, 1, 0, 4);
    }

    // Scaled latents [B·F, 4, h, w] to frames [B·F, 3, f·h, f·w] in [-1, 1] through the temporal decoder.
    private Tensor<T> DecodeLatents(Tensor<T> latents, int frames)
        => VaeDecoder.Forward(Engine.TensorMultiplyScalar(latents, NumOps.FromDouble(1.0 / _options.LatentScale)), frames);

    // Reverses the frames of each clip of [B·F, ...].
    private Tensor<T> ReverseFrames(Tensor<T> x, int frames)
    {
        int clips = x.Shape[0] / frames;
        var pieces = new List<Tensor<T>>();
        for (int b = 0; b < clips; b++)
            for (int f = frames - 1; f >= 0; f--) pieces.Add(Engine.TensorNarrow(x, 0, b * frames + f, 1));
        return Engine.TensorConcatenate(pieces.ToArray(), 0);
    }

    // The reference frame from the image super-resolution model (bicubic when none): [3, h, w] in [0, 1] to [3, sh, sw].
    private Tensor<T> UpscaleReference(Tensor<T> frame)
    {
        if (ReferenceUpscaler is { } upscaler) return upscaler(frame);
        int h = frame.Shape[1], w = frame.Shape[2];
        var up = Engine.Interpolate(Engine.Reshape(frame, new[] { 1, 3, h, w }), new[] { h * _options.ScaleFactor, w * _options.ScaleFactor },
            InterpolateMode.Bicubic, false);
        return Engine.Reshape(Engine.TensorClamp(up, NumOps.Zero, NumOps.One), up._shape.Skip(1).ToArray());
    }

    private Tensor<T> ToSigned(Tensor<T> x) => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));

    private Tensor<T> ToUnit(Tensor<T> x) => Engine.TensorMultiplyScalar(Engine.TensorAddScalar(x, NumOps.One), NumOps.FromDouble(0.5));

    // Low-resolution frames [B·F, 3, h, w] in [0, 1] bicubically brought to the output size, in [-1, 1].
    private Tensor<T> LowResUp(Tensor<T> low)
    {
        var up = Engine.Interpolate(low, new[] { low.Shape[2] * _options.ScaleFactor, low.Shape[3] * _options.ScaleFactor }, InterpolateMode.Bicubic, false);
        return ToSigned(Engine.TensorClamp(up, NumOps.Zero, NumOps.One));
    }

    // references: [B, 3, H, W] in [-1, 1]; lowRes: [B·F, 3, H, W] in [-1, 1].
    private PassCondition Condition(Tensor<T> references, Tensor<T> lowRes, int frames, Random random)
    {
        int b = references.Shape[0];
        var embedding = ImageEncoder.Forward(Engine, Layers, references);
        var context = SvdBlocks.RepeatInterleave(Engine, Engine.Reshape(embedding, new[] { b, 1, embedding.Shape[1] }), frames);
        // SVD's pipeline noises the conditioning image before encoding it and uses the posterior mode unscaled.
        var noised = Engine.TensorAdd(references,
            Engine.TensorMultiplyScalar(Gaussian(references.Shape.ToArray(), random), NumOps.FromDouble(_options.NoiseAugmentationStrength)));
        var reference = SvdBlocks.RepeatInterleave(Engine, EncodeLatent(noised), frames);
        return new PassCondition(context, reference, lowRes, frames);
    }

    private IReadOnlyList<double[]> AddedTimeIds(int clips)
    {
        var ids = new double[clips][];
        for (int i = 0; i < clips; i++) ids[i] = new double[] { _options.Fps - 1, _options.MotionBucketId, _options.NoiseAugmentationStrength };
        return ids;
    }

    /// <summary>
    /// The U-Net's raw output F for scaled latents <paramref name="z"/> at noise level σ: EDM's c_in on the input, c_noise =
    /// ln(σ) / 4 as the timestep, and the ControlNet's residuals.
    /// </summary>
    private Tensor<T> ModelOutput(Tensor<T> z, double sigma, PassCondition condition,
        Func<SvdSpatioTemporalTransformer<T>, IReadOnlyList<Func<Tensor<T>, Tensor<T>>>?>? temporalAttention)
    {
        int clips = z.Shape[0] / condition.Frames;
        var input = Engine.TensorConcatenate(new[] { Engine.TensorMultiplyScalar(z, NumOps.FromDouble(1.0 / Math.Sqrt(sigma * sigma + 1))), condition.Reference }, 1);
        var timesteps = Enumerable.Repeat(0.25 * Math.Log(sigma), clips).ToArray();
        var ids = AddedTimeIds(clips);
        var controlEmb = ControlNet.Embedding(Engine, Layers, timesteps, ids, condition.Frames);
        var (skips, middle) = ControlNet.Forward(Engine, Layers, input, condition.LowRes, controlEmb, condition.Context, condition.Frames);
        var emb = UNet.Embedding(Engine, Layers, timesteps, ids, condition.Frames);
        return UNet.Forward(Engine, Layers, input, emb, condition.Context, condition.Frames, skips, middle, temporalAttention);
    }

    // v-prediction under EDM (σ_data = 1): D = z / (σ² + 1) − σ / sqrt(σ² + 1) · F.
    private Tensor<T> Denoised(Tensor<T> z, Tensor<T> output, double sigma)
        => Engine.TensorAdd(Engine.TensorMultiplyScalar(z, NumOps.FromDouble(1.0 / (sigma * sigma + 1))),
            Engine.TensorMultiplyScalar(output, NumOps.FromDouble(-sigma / Math.Sqrt(sigma * sigma + 1))));

    // The F that would denoise z to z0: (z0 − c_skip·z) / c_out.
    private Tensor<T> OutputTarget(Tensor<T> z, Tensor<T> z0, double sigma)
        => Engine.TensorMultiplyScalar(Engine.TensorSubtract(z0, Engine.TensorMultiplyScalar(z, NumOps.FromDouble(1.0 / (sigma * sigma + 1)))),
            NumOps.FromDouble(-Math.Sqrt(sigma * sigma + 1) / sigma));

    // Records each temporal self-attention map of a forward pass, in execution order.
    private Func<SvdSpatioTemporalTransformer<T>, IReadOnlyList<Func<Tensor<T>, Tensor<T>>>?> CaptureMaps(List<Tensor<T>> maps)
        => transformer => transformer.TemporalBlocks
            .Select(block => (Func<Tensor<T>, Tensor<T>>)(n => block.TemporalSelfAttention.Forward(Engine, Layers, n, null, maps.Add)))
            .ToList();

    // The backward pass's temporal self-attention: the forward map rotated 180 degrees (A'[k−x, k−y] = A[x, y]) applied
    // with the backward pass's own value and output projections.
    private Func<SvdSpatioTemporalTransformer<T>, IReadOnlyList<Func<Tensor<T>, Tensor<T>>>?> ReplayMaps(IReadOnlyList<Tensor<T>> maps)
    {
        int next = 0;
        return transformer => transformer.TemporalBlocks.Select(_ =>
        {
            int index = next++;
            return (Func<Tensor<T>, Tensor<T>>)(n => RotatedMapAttention(n, maps[index], _backwardTemporal[index]));
        }).ToList();
    }

    /// <summary>The backward pass's temporal attention of temporal block <paramref name="index"/>: tokens [S, F, C] and
    /// the forward pass's map [S·heads, F, F].</summary>
    internal Tensor<T> BackwardTemporalAttention(int index, Tensor<T> x, Tensor<T> map) => RotatedMapAttention(x, map, _backwardTemporal[index]);

    private Tensor<T> RotatedMapAttention(Tensor<T> x, Tensor<T> map, (int Value, int Output) projections)
    {
        int s = x.Shape[0], f = x.Shape[1], heads = map.Shape[0] / s;
        var v = Layers[projections.Value].Forward(x);
        int inner = v.Shape[2], hd = inner / heads;
        var flip = new Tensor<T>(new[] { 1, f, f });
        for (int i = 0; i < f; i++) flip[i * f + (f - 1 - i)] = NumOps.One;
        var flips = Engine.TensorBroadcastTo(flip, new[] { s * heads, f, f });
        var rotated = Engine.TensorBatchMatMul(Engine.TensorBatchMatMul(flips, map), flips);
        var vh = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(v, new[] { s, f, heads, hd }), new[] { 0, 2, 1, 3 }), new[] { s * heads, f, hd });
        var merged = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(Engine.TensorBatchMatMul(rotated, vh), new[] { s, heads, f, hd }),
            new[] { 0, 2, 1, 3 }), new[] { s, f, inner });
        return Layers[projections.Output].Forward(merged);
    }

    /// <summary>
    /// SDEdit then Euler sampling: the clip's bicubic upscale is encoded, noised to the schedule's σ at the entry step and
    /// denoised to the end, each step averaging the forward pass with the motion-aligned backward pass when enabled.
    /// </summary>
    /// <param name="clips">[B, F, 3, h, w] low-resolution frames in [0, 1].</param>
    /// <returns>The scaled latents [B·F, 4, H / f, W / f].</returns>
    private Tensor<T> SampleLatents(Tensor<T> clips, int steps, Random random)
    {
        int b = clips.Shape[0], frames = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4];
        var flat = Engine.Reshape(clips, new[] { b * frames, 3, h, w });
        var lowUp = LowResUp(flat);
        Tensor<T> References(int frame)
        {
            var refs = new List<Tensor<T>>();
            for (int i = 0; i < b; i++)
            {
                var up = UpscaleReference(Engine.Reshape(Engine.TensorNarrow(flat, 0, i * frames + frame, 1), new[] { 3, h, w }));
                refs.Add(Engine.Reshape(ToSigned(up), new[] { 1, 3, up.Shape[1], up.Shape[2] }));
            }
            return Engine.TensorConcatenate(refs.ToArray(), 0);
        }

        var forward = Condition(References(0), lowUp, frames, random);
        var backward = _options.BidirectionalSampling ? Condition(References(frames - 1), ReverseFrames(lowUp, frames), frames, random) : null;
        var sigmas = Sigmas(steps);
        int start = SdeditStart(steps);
        var z0 = Engine.TensorMultiplyScalar(EncodeLatent(lowUp), NumOps.FromDouble(_options.LatentScale));
        var z = Engine.TensorAdd(z0, Engine.TensorMultiplyScalar(Gaussian(z0.Shape.ToArray(), random), NumOps.FromDouble(sigmas[start])));
        for (int i = start; i < steps; i++)
        {
            double sigma = sigmas[i];
            var maps = new List<Tensor<T>>();
            var denoised = Denoised(z, ModelOutput(z, sigma, forward, backward is null ? null : CaptureMaps(maps)), sigma);
            if (backward is not null)
            {
                var reversed = ReverseFrames(z, frames);
                var back = Denoised(reversed, ModelOutput(reversed, sigma, backward, ReplayMaps(maps)), sigma);
                denoised = Engine.TensorMultiplyScalar(Engine.TensorAdd(denoised, ReverseFrames(back, frames)), NumOps.FromDouble(0.5));
            }
            var slope = Engine.TensorMultiplyScalar(Engine.TensorSubtract(z, denoised), NumOps.FromDouble(1.0 / sigma));
            z = Engine.TensorAdd(z, Engine.TensorMultiplyScalar(slope, NumOps.FromDouble(sigmas[i + 1] - sigma)));
        }
        return z;
    }

    // Inference draws its noise from a fixed seed so the same clip always upscales to the same video.
    private Random InferenceRandom() => RandomHelper.CreateSeededRandom(_options.Seed ?? 0);

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the output size, in [0, 1].
    private Tensor<T> RunClips(Tensor<T> frames, int steps, Random random)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int b = clips.Shape[0], f = clips.Shape[1];
        var decoded = ToUnit(DecodeLatents(SampleLatents(clips, steps, random), f));
        var shape = new[] { b, f, 3, decoded.Shape[2], decoded.Shape[3] };
        var output = Engine.Reshape(decoded, shape);
        return batched ? output : Engine.Reshape(output, shape.Skip(1).ToArray());
    }

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        using var noGrad = new NoGradScope<T>();
        return RunClips(input, _options.NumInferenceSteps, InferenceRandom());
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input), _options.DecoderSamplingSteps, InferenceRandom());

    // Spectral normalization's power iteration runs on the host, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> PaperAdamW()
        => new(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ControlOptimizer()
        => _controlOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "controlnet");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BackwardOptimizer()
        => _backwardOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "backward");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DecoderOptimizer()
        => _decoderOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "decoder");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "discriminator");

    private Tensor<T> Mse(Tensor<T> a, Tensor<T> b)
    {
        var squared = Engine.TensorSquare(Engine.TensorSubtract(a, b));
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false);
    }

    // mmagic GANLoss('vanilla'): BCE with logits against an all-real or all-fake label, averaged over pixels.
    private Tensor<T> VanillaGanLoss(Tensor<T> logits, bool targetIsReal)
    {
        var softplus = Engine.Softplus(targetIsReal ? Engine.TensorNegate(logits) : logits);
        return Engine.ReduceMean(softplus, Enumerable.Range(0, softplus.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    // A diffusion training example: the low-resolution clip, the clean scaled latents, the noisy latents at σ and the
    // conditioning of both passes, built without gradients (the trained parts see them as data).
    private sealed class DiffusionExample
    {
        internal DiffusionExample(Tensor<T> clean, Tensor<T> noisy, double sigma, PassCondition forward, PassCondition backward)
        {
            Clean = clean;
            Noisy = noisy;
            Sigma = sigma;
            Forward = forward;
            Backward = backward;
        }

        internal Tensor<T> Clean { get; }
        internal Tensor<T> Noisy { get; }
        internal double Sigma { get; }
        internal PassCondition Forward { get; }
        internal PassCondition Backward { get; }
    }

    // low/high: [B, F, 3, h, w] and [B, F, 3, H, W] in [0, 1]. The ground-truth first (last) frame is the forward (backward)
    // pass's reference, which the image super-resolution model stands in for at inference.
    private DiffusionExample BuildExample(Tensor<T> low, Tensor<T> high, double sigma, Random random)
    {
        using var noGrad = new NoGradScope<T>();
        int b = low.Shape[0], frames = low.Shape[1];
        int H = high.Shape[3], W = high.Shape[4];
        var highFlat = ToSigned(Engine.Reshape(high, new[] { b * frames, 3, H, W }));
        var lowUp = LowResUp(Engine.Reshape(low, new[] { b * frames, 3, low.Shape[3], low.Shape[4] }));
        Tensor<T> Frame(int f) => Engine.TensorConcatenate(Enumerable.Range(0, b)
            .Select(i => Engine.TensorNarrow(highFlat, 0, i * frames + f, 1)).ToArray(), 0);
        var forward = Condition(Frame(0), lowUp, frames, random);
        var backward = Condition(Frame(frames - 1), ReverseFrames(lowUp, frames), frames, random);
        var clean = Engine.TensorMultiplyScalar(EncodeLatent(highFlat), NumOps.FromDouble(_options.LatentScale));
        var noisy = Engine.TensorAdd(clean, Engine.TensorMultiplyScalar(Gaussian(clean.Shape.ToArray(), random), NumOps.FromDouble(sigma)));
        return new DiffusionExample(clean, noisy, sigma, forward, backward);
    }

    // EDM's training noise: log σ ~ N(TrainingSigmaMean, TrainingSigmaStd²).
    private double SampleSigma(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Exp(_options.TrainingSigmaMean + _options.TrainingSigmaStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    // The ControlNet stage's loss: EDM v-prediction MSE (λ(σ)·c_out² = 1) of the forward pass.
    private Tensor<T> ForwardLoss(DiffusionExample e)
        => Mse(ModelOutput(e.Noisy, e.Sigma, e.Forward, null), OutputTarget(e.Noisy, e.Clean, e.Sigma));

    // The backward stage's loss: the reversed clip through the backward pass, with the forward pass's maps rotated.
    private Tensor<T> BackwardLoss(DiffusionExample e)
    {
        var maps = new List<Tensor<T>>();
        using (new NoGradScope<T>()) ModelOutput(e.Noisy, e.Sigma, e.Forward, CaptureMaps(maps));
        int frames = e.Forward.Frames;
        var reversed = ReverseFrames(e.Noisy, frames);
        return Mse(ModelOutput(reversed, e.Sigma, e.Backward, ReplayMaps(maps)), OutputTarget(reversed, ReverseFrames(e.Clean, frames), e.Sigma));
    }

    private (Tensor<T> Low, Tensor<T> High) AsClips(Tensor<T> input, Tensor<T> target)
    {
        var low = PreprocessFrames(input);
        var high = NormalizeFrames(target);
        return low.Rank == 5
            ? (low, high)
            : (Engine.Reshape(low, new[] { 1 }.Concat(low._shape).ToArray()), Engine.Reshape(high, new[] { 1 }.Concat(high._shape).ToArray()));
    }

    /// <inheritdoc />
    /// <remarks>
    /// The paper's three stages in turn: the ControlNet with the U-Net frozen; the backward pass's temporal value and output
    /// projections with everything else frozen; then the VAE decoder on L2, perceptual and GAN losses of 8-step samples,
    /// followed by the discriminator on the real clip and the decoded one.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        int stage = TrainingStage;
        _trainingSteps++;
        _lastFake = null;
        var (low, high) = AsClips(input, expectedOutput);
        if (stage < 2)
        {
            var example = BuildExample(low, high, SampleSigma(_random), _random);
            return new[]
            {
                stage == 0
                    ? new TrainingObjective<T>(new[] { input, expectedOutput }, _ => ForwardLoss(example))
                    {
                        Optimizer = ControlOptimizer(),
                        Parameters = () => ControlNetParameters,
                    }
                    : new TrainingObjective<T>(new[] { input, expectedOutput }, _ => BackwardLoss(example))
                    {
                        Optimizer = BackwardOptimizer(),
                        Parameters = () => BackwardParameters,
                    },
            };
        }

        Tensor<T> latents;
        using (new NoGradScope<T>()) latents = SampleLatents(low, _options.DecoderSamplingSteps, _random);
        int frames = low.Shape[1];
        var real = Engine.Reshape(high, new[] { high.Shape[0] * frames, 3, high.Shape[3], high.Shape[4] });
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, _ =>
            {
                var decoded = ToUnit(DecodeLatents(latents, frames));
                _lastFake = Engine.StopGradient(decoded);
                var loss = Mse(decoded, real);
                if (_vgg is not null) loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, decoded, real), _options.PerceptualWeight));
                return Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, decoded, training: true), true), _options.GANWeight));
            })
            {
                Optimizer = DecoderOptimizer(),
                Parameters = () => DecoderParameters,
            },
            new TrainingObjective<T>(
                () => new[] { real, _lastFake ?? throw new InvalidOperationException("The decoder step did not run.") },
                data => Engine.TensorAdd(
                    VanillaGanLoss(Discriminator.Run(Engine, Layers, data[0], training: true), true),
                    VanillaGanLoss(Discriminator.Run(Engine, Layers, data[1], training: true), false)))
            {
                Optimizer = DiscriminatorOptimizer(),
                Parameters = () => DiscriminatorParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.DiffusionDenoising;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>
    /// The current stage's loss as a deterministic quadrature: noise from a fixed seed at log σ on the training
    /// distribution's 1/6, 3/6 and 5/6 quantiles for the diffusion stages, and the decoder's L2 loss on fixed-seed samples
    /// for the last.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var (low, high) = AsClips(input, target);
        int stage = TrainingStage;
        if (stage == 2)
        {
            int frames = low.Shape[1];
            var decoded = ToUnit(DecodeLatents(SampleLatents(low, _options.DecoderSamplingSteps, InferenceRandom()), frames));
            return Mse(decoded, Engine.Reshape(high, new[] { high.Shape[0] * frames, 3, high.Shape[3], high.Shape[4] }))[0];
        }

        double total = 0;
        foreach (double z in new[] { -0.9674, 0.0, 0.9674 })
        {
            var random = RandomHelper.CreateSeededRandom(1729);
            var example = BuildExample(low, high, Math.Exp(_options.TrainingSigmaMean + _options.TrainingSigmaStd * z), random);
            total += NumOps.ToDouble((stage == 0 ? ForwardLoss(example) : BackwardLoss(example))[0]);
        }
        return NumOps.FromDouble(total / 3);
    }

    /// <inheritdoc />
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        base.OnTrainingStepEnded(succeeded);
        _lastFake = null;
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? ControlOptimizer() : base.GetOrCreateBaseOptimizer();

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
            Name = _useNativeMode ? "DAMVSR-Native" : "DAMVSR-ONNX",
            Description = "DAM-VSR appearance and motion disentangled video super-resolution on Stable Video Diffusion (Kong et al. 2025)",
            Complexity = _options.UNetChannels.Length
        };
        m.AdditionalInfo["UNetChannels"] = string.Join(",", _options.UNetChannels);
        m.AdditionalInfo["NumInferenceSteps"] = _options.NumInferenceSteps.ToString();
        m.AdditionalInfo["SdeditStrength"] = _options.SdeditStrength.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["BidirectionalSampling"] = _options.BidirectionalSampling.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DAMVSR<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
