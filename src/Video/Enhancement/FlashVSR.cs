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
/// FlashVSR: real-time diffusion-based streaming video super-resolution, distilled to one step from Wan 2.1.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FlashVSR (Zhuang et al. 2025) conditions Wan 2.1-1.3B on the low-resolution clip through a light projection whose
/// tokens are added to the DiT's, not through the VAE encoder. One step at timestep 1000 maps noise to the latent
/// (z = ε − v), and the tiny conditional decoder turns it into frames using the LR clip as well. The release streams two
/// latent frames at a time against a cache of the previous three chunks, with attention restricted to the top-scoring
/// (2, 8, 8)-token blocks inside a locality window; evaluated over the whole clip with the equivalent block-causal mask,
/// as here, the result is the same. Colors are matched to the LR upscale per frame (AdaIN).
/// </para>
/// <para>
/// Training follows Sec. 3-4 in four stages: the tiny decoder alone (L2 and LPIPS against the ground truth and against
/// Wan's decoder); flow matching with full attention, which becomes the frozen teacher; flow matching with the sparse
/// causal attention; and distribution-matching distillation of the one-step model against the teacher and a fake-score
/// critic, with L2 and LPIPS on the decoded frames. The DiT trains through LoRA adapters, the LR projection fully.
/// </para>
/// <para>
/// <b>For Beginners:</b> FlashVSR sharpens video in real time. It starts from a large video generation model, teaches it
/// to produce a sharp frame in a single pass, and limits each pass to recent frames and nearby regions so it keeps up
/// with a live stream.
/// </para>
/// <para>
/// Frames are padded rather than dropped: the release repeats the last frame four times and keeps the largest 8n + 1
/// frames, cropping the rest, and crops each side to a multiple of 128; here the clip is padded to the next valid length
/// and size and the output cropped back, so every input frame and pixel has an output. The fixed prompt's T5 embedding is
/// part of the checkpoint and held frozen; untrained, it is zero. The image half of the paper's Stage 1 data is left out:
/// the released LR projection emits no token for a lone frame.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution",
    "https://arxiv.org/abs/2510.12747",
    Year = 2025,
    Authors = "Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0.01, Component = "generator",
                Source = "Zhuang et al. 2025, Sec. 4.1: AdamW with a learning rate of 1e-5 and weight decay of 0.01 for every stage.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0.01, Component = "critic",
                Source = "Zhuang et al. 2025, Sec. 4.1: the same AdamW for the distillation's fake-score network.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0.01, Component = "decoder",
                Source = "Zhuang et al. 2025, Sec. 3.4 trains the tiny decoder separately without naming its optimizer; "
                        + "the paper's AdamW setting is used.")]
public partial class FlashVSR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private const int Student = 0, Teacher = 1, Fake = 2;
    private const int FirstChunkFrames = 6;

    private readonly FlashVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _criticOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _decoderOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private WanDiT<T>? _dit;
    private int _promptEmbedding;
    private readonly FlashVsrLqProjection<T>?[] _projections = new FlashVsrLqProjection<T>?[WanLoraLinear<T>.Sets];
    private FlashVsrTcDecoder<T>? _decoder;
    private WanVideoVae<T>? _vae;
    private LpipsVggBlock<T>? _lpips;
    private double[] _sigmas = Array.Empty<double>();
    private Random _random;

    /// <summary>Creates a FlashVSR model in ONNX inference mode.</summary>
    public FlashVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, FlashVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new FlashVSROptions() : new FlashVSROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a FlashVSR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's and Wan 2.1-1.3B's defaults when null.</param>
    /// <param name="optimizer">The DiT's optimizer; the paper's AdamW when null. The critic and the decoder always use it.</param>
    public FlashVSR(NeuralNetworkArchitecture<T> architecture, FlashVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _generatorOptimizer = optimizer;
        _options = options is null ? new FlashVSROptions() : new FlashVSROptions(options);
        Options = _options;
        var o = _options;
        if (o.ModelDim < 1 || o.NumHeads < 1 || o.ModelDim % o.NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ModelDim must split into NumHeads heads.");
        int headDim = o.ModelDim / o.NumHeads;
        if (headDim % 2 != 0 || (headDim / 3) % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The head width must be even with an even third for the 3D rotary embedding.");
        if (o.NumLayers < 1 || o.FfnDim < 1 || o.FrequencyDim < 2 || o.TextDim < 1 || o.TextLength < 1 || o.LoraRank < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The DiT's sizes must be positive.");
        if (o.LqProjectionLayers < 1 || o.LqProjectionLayers > o.NumLayers || o.LqHiddenDim1 < 1 || o.LqHiddenDim2 < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The LR projection needs widths and between one and NumLayers layers.");
        if (o.DecoderChannels is not { Length: 4 } || o.DecoderChannels.Any(c => c < 1))
            throw new ArgumentOutOfRangeException(nameof(options), "DecoderChannels must hold four positive widths.");
        if (o.VaeDimMultipliers is not { Length: 4 } || o.VaeDimMultipliers.Any(m => m < 1) || o.VaeDim < 1 || o.VaeResidualBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Wan's VAE needs four stages (x8 in space, x4 in time).");
        if (o.ScaleFactor < 1 || o.SparseRatio <= 0 || o.KvCacheChunks < 1 || o.LocalRange < 1 || o.NumTrainTimesteps < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor, the sparse attention and the schedule must be valid.");
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
        state.DeclareDoubleVector(nameof(FlashVSR<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
            restored => _trainingSteps = restored is { Length: > 0 } ? (int)restored[0] : 0);
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        var frozen = AiDotNet.Models.Parameters.ParameterSlotRole.Frozen;
        // Wan's VAE only supplies the training latents and the decoder's second target; it is never trained.
        RegisterParameterComponent("FlashVSR.Vae", _vae, frozen);
        RegisterParameterComponent("FlashVSR.LqProjection", _projections[Student]);
        RegisterParameterComponent("FlashVSR.TeacherLqProjection", _projections[Teacher], frozen);
        RegisterParameterComponent("FlashVSR.CriticLqProjection", _projections[Fake]);
        RegisterParameterComponent("FlashVSR.Decoder", _decoder);
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
        _dit = new WanDiT<T>(Layers, WanVideoVae<T>.LatentChannels, o.ModelDim, o.FfnDim, o.NumHeads, o.NumLayers, o.FrequencyDim,
            o.LoraRank, o.NormEpsilon);
        _promptEmbedding = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.TextLength * o.TextDim, 0.0));
        for (int s = 0; s < WanLoraLinear<T>.Sets; s++)
            _projections[s] = new FlashVsrLqProjection<T>(o.LqHiddenDim1, o.LqHiddenDim2, o.ModelDim, o.LqProjectionLayers);
        _decoder = new FlashVsrTcDecoder<T>(o.DecoderChannels, WanVideoVae<T>.LatentChannels);
        _vae = new WanVideoVae<T>(o.VaeDim, o.VaeDimMultipliers, o.VaeResidualBlocks);
        // A zero LPIPS weight leaves the VGG16 out entirely rather than carrying it unused.
        if (o.LpipsWeight > 0) _lpips = new LpipsVggBlock<T>(Layers);
        _sigmas = Sigmas(o);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        // One dense pass per adapter set on a single-token latent sizes every linear map.
        var latent = new Tensor<T>(new[] { 1, WanVideoVae<T>.LatentChannels, 1, 2, 2 });
        for (int s = 0; s < WanLoraLinear<T>.Sets; s++)
            Dit.Forward(Engine, Layers, latent, new[] { 1000.0 }, Prompt(1), null, s, WanAttentionMode.Dense, null);
        var image = new Tensor<T>(new[] { 1, 3, 32, 32 });
        _lpips?.Loss(Engine, Layers, image, image);
        MarkLayerShapesResolved();
    }

    private WanDiT<T> Dit => _dit ?? throw new InvalidOperationException("FlashVSR's native layers are not initialized.");

    private FlashVsrLqProjection<T> Projection(int set)
        => _projections[set] ?? throw new InvalidOperationException("FlashVSR's native layers are not initialized.");

    private FlashVsrTcDecoder<T> Decoder => _decoder ?? throw new InvalidOperationException("FlashVSR's native layers are not initialized.");

    private WanVideoVae<T> Vae => _vae ?? throw new InvalidOperationException("FlashVSR's native layers are not initialized.");

    /// <summary>The trainable weights of one network: its LoRA adapters and its LR projection.</summary>
    private IReadOnlyList<Tensor<T>> NetworkParameters(int set) => _usingCustomLayers || _dit is null
        ? Array.Empty<Tensor<T>>()
        : TapeTrainingStep<T>.CollectParameters(_dit.Adapter(set).Select(i => Layers[i]))
            .Concat(Projection(set).GetParameterStateChunks().Select(chunk => chunk.Tensor)).ToList();

    /// <summary>The one-step network's LoRA adapters and LR projection.</summary>
    internal IReadOnlyList<Tensor<T>> GeneratorParameters => NetworkParameters(Student);

    /// <summary>The frozen teacher's LoRA adapters and LR projection.</summary>
    internal IReadOnlyList<Tensor<T>> TeacherParameters => NetworkParameters(Teacher);

    /// <summary>The fake-score critic's LoRA adapters and LR projection.</summary>
    internal IReadOnlyList<Tensor<T>> CriticParameters => NetworkParameters(Fake);

    /// <summary>The tiny conditional decoder's weights.</summary>
    internal IReadOnlyList<Tensor<T>> DecoderParameters => _decoder is null
        ? Array.Empty<Tensor<T>>()
        : _decoder.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    /// <summary>The training stage the next step runs: 0 decoder, 1 teacher, 2 sparse causal, 3 distillation.</summary>
    internal int TrainingStage
    {
        get
        {
            int step = _trainingSteps, o1 = _options.DecoderIterations, o2 = o1 + _options.TeacherIterations, o3 = o2 + _options.SparseIterations;
            return step < o1 ? 0 : step < o2 ? 1 : step < o3 ? 2 : 3;
        }
    }

    #endregion

    #region One-step pipeline

    /// <summary>
    /// The flow-matching σ of each training timestep (DiffSynth's <c>FlowMatchScheduler</c> with Wan's shift and an extra
    /// step): N values from 1 down to 0 excluding the last, each shifted to s·σ / (1 + (s − 1)·σ). The timestep is 1000·σ.
    /// </summary>
    internal static double[] Sigmas(FlashVSROptions options) => Sigmas(options.NumTrainTimesteps, options.SigmaShift);

    /// <summary>The same σ grid for a schedule of <paramref name="n"/> steps and shift <paramref name="shift"/>.</summary>
    internal static double[] Sigmas(int n, double shift)
    {
        var sigmas = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sigma = 1.0 - i / (double)n;
            sigmas[i] = shift * sigma / (1.0 + (shift - 1.0) * sigma);
        }
        return sigmas;
    }

    private Tensor<T> Prompt(int batch)
    {
        var prompt = Engine.Reshape(RealViformerBlocks.Read(Layers, _promptEmbedding), new[] { 1, _options.TextLength, _options.TextDim });
        return Engine.TensorBroadcastTo(prompt, new[] { batch, _options.TextLength, _options.TextDim });
    }

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

    // The LR length the pipeline runs: the smallest 8m + 1 that leaves at least the input's frames after the four the
    // decoder's trim consumes, and at least the first streamed step's six latents.
    private static int PaddedLength(int frames)
    {
        int needed = Math.Max(frames + 4, 4 * FirstChunkFrames + 1);
        return (needed - 1 + 7) / 8 * 8 + 1;
    }

    // Repeats the last frame of [B, F, ...] until there are `length` frames.
    private Tensor<T> RepeatLast(Tensor<T> clips, int length)
    {
        int f = clips.Shape[1];
        if (f >= length) return clips;
        var target = clips.Shape.ToArray();
        target[1] = length - f;
        return Engine.TensorConcatenate(new[] { clips, Engine.TensorBroadcastTo(Engine.TensorNarrow(clips, 1, f - 1, 1), target) }, 1);
    }

    // The LR side unit: s times it is a multiple of 128 (the (2, 8, 8) windows at the VAE's x8 and the DiT's x2).
    private int LowUnit
    {
        get
        {
            int a = 128, b = _options.ScaleFactor;
            while (b != 0) (a, b) = (b, a % b);
            return 128 / a;
        }
    }

    // Replicate-pads the right and bottom of [B, F, 3, h, w] to multiples of unit.
    private Tensor<T> PadSides(Tensor<T> clips, int unit)
    {
        int h = clips.Shape[3], w = clips.Shape[4];
        int padH = (unit - h % unit) % unit, padW = (unit - w % unit) % unit;
        if (padH == 0 && padW == 0) return clips;
        return Engine.PadNd(clips, new[] { 0, padW, 0, padH }, PadMode.Replicate, NumOps.Zero);
    }

    // LR clips [B, F, 3, h, w] in [0, 1] to the bicubic upscale [B, 3, F, s·h, s·w] in [-1, 1] (the release's LQ video).
    private Tensor<T> Upsample(Tensor<T> clips)
    {
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var up = Engine.Interpolate(Engine.Reshape(clips, new[] { b * f, 3, h, w }), new[] { h * s, w * s }, InterpolateMode.Bicubic, false);
        return Engine.TensorPermute(Engine.Reshape(ToSigned(up), new[] { b, f, 3, h * s, w * s }), new[] { 0, 2, 1, 3, 4 });
    }

    private WanSparsePlan SparsePlan(int frames, int height, int width)
        => new(frames, height, width, _options.SparseRatio * 768 * 1280 / (256.0 * height * width), _options.KvCacheChunks, _options.LocalRange);

    // The network's velocity for latents [B, 16, n, h, w] at per-sample timesteps, conditioned on the LQ video [B, 3, 4n + 1, H, W].
    private Tensor<T> Velocity(Tensor<T> latents, IReadOnlyList<double> timesteps, Tensor<T> lq, int set, WanAttentionMode mode)
        => Dit.Forward(Engine, Layers, latents, timesteps, Prompt(latents.Shape[0]), Projection(set).Forward(lq), set, mode, SparsePlan);

    // The one step: z = ε − v_θ(ε, 1000) for the LQ video [B, 3, 4n + 1, H, W].
    private Tensor<T> OneStep(Tensor<T> lq, Random random)
    {
        int b = lq.Shape[0], n = (lq.Shape[2] - 1) / 4, f = Vae.SpatialFactor;
        var noise = Gaussian(new[] { b, WanVideoVae<T>.LatentChannels, n, lq.Shape[3] / f, lq.Shape[4] / f }, random);
        var v = Velocity(noise, Enumerable.Repeat(1000.0, b).ToArray(), lq, Student, WanAttentionMode.SparseCausal);
        return Engine.TensorSubtract(noise, v);
    }

    // The tiny decoder on latents [B, 16, n, h, w] with the LQ video's first 4n − 3 frames: [B, 3, 4n − 3, H, W] in [-1, 1].
    private Tensor<T> Decode(Tensor<T> latents, Tensor<T> lq)
        => ToSigned(Decoder.Forward(latents, Engine.TensorNarrow(lq, 2, 0, 4 * latents.Shape[2] - 3)));

    // AdaIN per frame and channel against the LQ video (unbiased = False, ε 1e-5 on the variance), clamped to [-1, 1].
    private Tensor<T> ColorFix(Tensor<T> frames, Tensor<T> lq)
    {
        var shape = frames.Shape.ToArray();
        var axes = new[] { 3, 4 };
        Tensor<T> Mean(Tensor<T> x) => Engine.TensorBroadcastTo(Engine.ReduceMean(x, axes, true), shape);
        Tensor<T> Std(Tensor<T> x, Tensor<T> mean)
            => Engine.TensorSqrt(Engine.TensorAddScalar(Mean(Engine.TensorSquare(Engine.TensorSubtract(x, mean))), NumOps.FromDouble(1e-5)));
        var mean = Mean(frames);
        var styleMean = Mean(lq);
        var normalized = Engine.TensorDivide(Engine.TensorSubtract(frames, mean), Std(frames, mean));
        var output = Engine.TensorAdd(Engine.TensorMultiply(normalized, Std(lq, styleMean)), styleMean);
        return Engine.TensorClamp(output, NumOps.FromDouble(-1.0), NumOps.One);
    }

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the output size, in [0, 1].
    private Tensor<T> RunClips(Tensor<T> frames, Random random)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var lq = Upsample(PadSides(RepeatLast(clips, PaddedLength(f)), LowUnit));
        var video = Decode(OneStep(lq, random), lq);
        var start = new[] { 0, 0, 0, 0, 0 };
        var size = new[] { b, 3, f, h * s, w * s };
        video = Engine.TensorSlice(video, start, size);
        if (_options.ColorFix) video = ColorFix(video, Engine.TensorSlice(lq, start, size));
        var output = Engine.TensorPermute(ToUnit(video), new[] { 0, 2, 1, 3, 4 });
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

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> PaperAdamW()
        => new(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GeneratorOptimizer()
        => _generatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CriticOptimizer()
        => _criticOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "critic");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DecoderOptimizer()
        => _decoderOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(), "decoder");

    private Tensor<T> Mse(Tensor<T> a, Tensor<T> b)
    {
        var squared = Engine.TensorSquare(Engine.TensorSubtract(a, b));
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false);
    }

    // LPIPS over the frames of [B, 3, F, H, W] videos in [-1, 1], weighted by λ; zero when the weight is.
    private Tensor<T>? Perceptual(Tensor<T> video, Tensor<T> target)
    {
        if (_lpips is null) return null;
        int b = video.Shape[0], f = video.Shape[2], h = video.Shape[3], w = video.Shape[4];
        Tensor<T> Frames(Tensor<T> v) => Engine.Reshape(Engine.TensorPermute(v, new[] { 0, 2, 1, 3, 4 }), new[] { b * f, 3, h, w });
        return Engine.TensorMultiplyScalar(_lpips.Loss(Engine, Layers, Frames(video), Frames(target)), NumOps.FromDouble(_options.LpipsWeight));
    }

    // L2 + λ·LPIPS (Eqs. 2 and 3).
    private Tensor<T> PixelLoss(Tensor<T> video, Tensor<T> target)
    {
        var loss = Mse(video, target);
        var perceptual = Perceptual(video, target);
        return perceptual is null ? loss : Engine.TensorAdd(loss, perceptual);
    }

    /// <summary>One training clip prepared as the pipeline runs it.</summary>
    private sealed class Batch
    {
        internal Batch(Tensor<T> lq, Tensor<T> high, Tensor<T> latents)
        {
            Lq = lq;
            High = high;
            Latents = latents;
        }

        /// <summary>The LQ video [B, 3, 4n + 1, H, W] in [-1, 1].</summary>
        internal Tensor<T> Lq { get; }

        /// <summary>The ground truth [B, 3, 4n − 3, H, W] in [-1, 1].</summary>
        internal Tensor<T> High { get; }

        /// <summary>Its Wan VAE latents [B, 16, n, H / 8, W / 8].</summary>
        internal Tensor<T> Latents { get; }
    }

    private Batch Prepare(Tensor<T> input, Tensor<T> target)
    {
        var low = PreprocessFrames(input);
        var high = NormalizeFrames(target);
        if (low.Rank == 4)
        {
            low = Engine.Reshape(low, new[] { 1 }.Concat(low._shape).ToArray());
            high = Engine.Reshape(high, new[] { 1 }.Concat(high._shape).ToArray());
        }
        using var noGrad = new NoGradScope<T>();
        int length = PaddedLength(low.Shape[1]);
        var lq = Upsample(PadSides(RepeatLast(low, length), LowUnit));
        var truth = PadSides(RepeatLast(high, length - 4), LowUnit * _options.ScaleFactor);
        var video = Engine.TensorPermute(ToSigned(truth), new[] { 0, 2, 1, 3, 4 });
        return new Batch(lq, video, Vae.Encode(video));
    }

    // Flow matching at a random training timestep: x_t = (1 − σ)·z + σ·ε against v = ε − z.
    private Tensor<T> FlowMatchingLoss(Tensor<T> latents, Tensor<T> lq, int set, WanAttentionMode mode, Random random)
    {
        int b = latents.Shape[0];
        var sigmas = Enumerable.Range(0, b).Select(_ => _sigmas[random.Next(_sigmas.Length)]).ToArray();
        var noise = Gaussian(latents.Shape.ToArray(), random);
        var noisy = Mix(latents, noise, sigmas);
        var target = Engine.TensorSubtract(noise, latents);
        return Mse(Velocity(noisy, sigmas.Select(s => 1000.0 * s).ToArray(), lq, set, mode), target);
    }

    // (1 − σ_b)·z + σ_b·ε per sample.
    private Tensor<T> Mix(Tensor<T> latents, Tensor<T> noise, IReadOnlyList<double> sigmas)
    {
        var shape = latents.Shape.ToArray();
        var sigma = SampleScalars(sigmas, shape);
        var keep = Engine.TensorAddScalar(Engine.TensorNegate(sigma), NumOps.One);
        return Engine.TensorAdd(Engine.TensorMultiply(latents, keep), Engine.TensorMultiply(noise, sigma));
    }

    private Tensor<T> SampleScalars(IReadOnlyList<double> values, int[] shape)
    {
        var column = new Tensor<T>(new[] { values.Count, 1, 1, 1, 1 }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
        return Engine.TensorBroadcastTo(column, shape);
    }

    // Stage 0, Eq. 3: the tiny decoder against the ground truth and against Wan's decoder on the same latents.
    private Tensor<T> DecoderLoss(Batch batch, Tensor<T> wanFrames)
    {
        var decoded = Decode(batch.Latents, batch.Lq);
        return Engine.TensorAdd(PixelLoss(decoded, batch.High), PixelLoss(decoded, wanFrames));
    }

    /// <summary>
    /// Eq. 2's distribution-matching term (Yin et al. 2024): at a random timestep in [0.02, 0.98] of the schedule, the
    /// teacher's and the critic's denoised estimates of the noised prediction give the gradient (fake − real) / mean|z −
    /// real|, applied as 0.5·MSE(z, sg(z − gradient)).
    /// </summary>
    private Tensor<T> DistributionMatching(Tensor<T> prediction, Tensor<T> lq, Random random)
    {
        int b = prediction.Shape[0], n = _sigmas.Length;
        int lo = (int)(0.02 * n), hi = Math.Max(lo + 1, (int)(0.98 * n));
        var sigmas = Enumerable.Range(0, b).Select(_ => _sigmas[lo + random.Next(hi - lo)]).ToArray();
        var shape = prediction.Shape.ToArray();
        Tensor<T> target;
        using (new NoGradScope<T>())
        {
            var noisy = Mix(prediction, Gaussian(shape, random), sigmas);
            var timesteps = sigmas.Select(s => 1000.0 * s).ToArray();
            var sigma = SampleScalars(sigmas, shape);
            var real = Engine.TensorSubtract(noisy, Engine.TensorMultiply(sigma, Velocity(noisy, timesteps, lq, Teacher, WanAttentionMode.Dense)));
            var fake = Engine.TensorSubtract(noisy, Engine.TensorMultiply(sigma, Velocity(noisy, timesteps, lq, Fake, WanAttentionMode.Dense)));
            var axes = new[] { 1, 2, 3, 4 };
            var scale = Engine.TensorBroadcastTo(Engine.TensorMax(
                Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(prediction, real)), axes, true), NumOps.FromDouble(1e-12)), shape);
            target = Engine.TensorSubtract(prediction, Engine.TensorDivide(Engine.TensorSubtract(fake, real), scale));
        }
        return Engine.TensorMultiplyScalar(Mse(prediction, target), NumOps.FromDouble(0.5));
    }

    // Stage 3's generator loss: distribution matching on the one-step latent plus L2 and LPIPS on its decoded frames. The
    // paper decodes two random latents an iteration to save memory; all are decoded here, the loss that estimates.
    private Tensor<T> GeneratorLoss(Batch batch, Random random)
    {
        var prediction = OneStep(batch.Lq, random);
        var loss = DistributionMatching(prediction, batch.Lq, random);
        return Engine.TensorAdd(loss, PixelLoss(Decode(prediction, batch.Lq), batch.High));
    }

    // Copies one network's adapters and LR projection over another's.
    private void CopyNetwork(int from, int to)
    {
        var source = Dit.Adapter(from).ToList();
        var destination = Dit.Adapter(to).ToList();
        for (int i = 0; i < source.Count; i++) Layers[destination[i]].SetParameters(Layers[source[i]].GetParameters());
        Projection(to).SetParameters(Projection(from).GetParameters());
    }

    /// <inheritdoc />
    /// <remarks>
    /// The paper's stages in turn: the tiny decoder; full-attention flow matching (the network then frozen as the teacher);
    /// sparse causal flow matching; and distillation, where the one-step network and the fake-score critic (initialized
    /// from the teacher) train in alternation.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        // Train on one NumFrames-frame clip of a longer sequence, as the paper's data pipeline does.
        (input, expectedOutput) = TrainingClip(input, expectedOutput);
        int stage = TrainingStage;
        _trainingSteps++;
        var o = _options;
        // The full-attention network becomes the frozen teacher as Stage 2 starts, and the critic starts from the teacher.
        int teacherEnd = o.DecoderIterations + o.TeacherIterations;
        if (stage >= 2 && _trainingSteps == teacherEnd + 1) CopyNetwork(Student, Teacher);
        if (stage == 3 && _trainingSteps == teacherEnd + o.SparseIterations + 1) CopyNetwork(Teacher, Fake);
        var batch = Prepare(input, expectedOutput);
        var sources = new[] { input, expectedOutput };
        var random = RandomHelper.CreateSeededRandom(_random.Next());

        if (stage == 0)
        {
            Tensor<T> wan;
            using (new NoGradScope<T>()) wan = Vae.Decode(batch.Latents);
            return new[]
            {
                new TrainingObjective<T>(sources, _ => DecoderLoss(batch, wan)) { Optimizer = DecoderOptimizer(), Parameters = () => DecoderParameters },
            };
        }
        if (stage < 3)
        {
            var mode = stage == 1 ? WanAttentionMode.Dense : WanAttentionMode.SparseCausal;
            return new[]
            {
                new TrainingObjective<T>(sources, _ => FlowMatchingLoss(batch.Latents, batch.Lq, Student, mode, random))
                {
                    Optimizer = GeneratorOptimizer(),
                    Parameters = () => GeneratorParameters,
                },
            };
        }

        Tensor<T> detached;
        using (new NoGradScope<T>()) detached = OneStep(batch.Lq, RandomHelper.CreateSeededRandom(random.Next()));
        var criticRandom = RandomHelper.CreateSeededRandom(random.Next());
        return new[]
        {
            new TrainingObjective<T>(sources, _ => GeneratorLoss(batch, random)) { Optimizer = GeneratorOptimizer(), Parameters = () => GeneratorParameters },
            new TrainingObjective<T>(sources, _ => FlowMatchingLoss(detached, batch.Lq, Fake, WanAttentionMode.Dense, criticRandom))
            {
                Optimizer = CriticOptimizer(),
                Parameters = () => CriticParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.DiffusionDenoising;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The current stage's loss with every random draw from a fixed seed.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var batch = Prepare(input, target);
        var random = RandomHelper.CreateSeededRandom(1729);
        return TrainingStage switch
        {
            0 => DecoderLoss(batch, Vae.Decode(batch.Latents))[0],
            1 => FlowMatchingLoss(batch.Latents, batch.Lq, Student, WanAttentionMode.Dense, random)[0],
            2 => FlowMatchingLoss(batch.Latents, batch.Lq, Student, WanAttentionMode.SparseCausal, random)[0],
            _ => GeneratorLoss(batch, random)[0],
        };
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? GeneratorOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    // Each step draws its batch on the host (the sigmas and the noise of the flow-matching and distillation losses,
    // the student's detached one-step sample), which a compiled plan would freeze at its first step.
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
            Name = _useNativeMode ? "FlashVSR-Native" : "FlashVSR-ONNX",
            Description = "FlashVSR one-step streaming video super-resolution on Wan 2.1 (Zhuang et al. 2025)",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumLayers"] = _options.NumLayers.ToString();
        m.AdditionalInfo["LoraRank"] = _options.LoraRank.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FlashVSR<T>));
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
