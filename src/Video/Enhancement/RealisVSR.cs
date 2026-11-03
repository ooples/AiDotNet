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
/// RealisVSR: detail-enhanced diffusion for real-world 4K video super-resolution, a Consistency-Preserved ControlNet on Wan 2.1.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RealisVSR (Zhao et al. 2025) keeps Wan 2.1-1.3B frozen and conditions it on the low-resolution clip through a ControlNet
/// branch that sees only the condition: the Wan VAE latents of the bicubic upscale, patch-embedded and passed through a few
/// Wan blocks that share the main network's prompt, time modulation and rotary positions. After main block i, the branch's
/// feature ⌊i / r⌋ scaled by a learned γ is added (Eq. 1). Sampling integrates the rectified flow from noise with Euler steps.
/// </para>
/// <para>
/// Training (Sec. 3.3) updates only the branch with the High-Frequency Rectified Diffusion Loss: the flow-matching MSE on the
/// velocity plus a weighted Haar wavelet loss (Eq. 4) and a HOG loss (Eq. 5) on the same velocity, so edges and textures weigh
/// more than the flat low-frequency content the MSE is dominated by.
/// </para>
/// <para>
/// <b>For Beginners:</b> RealisVSR takes a low-quality video and uses a large pretrained video generator to produce a sharp,
/// high-resolution version. A small side network reads the low-quality frames and steers the generator, and its training
/// loss puts extra weight on fine detail such as text and texture so those are not smoothed away.
/// </para>
/// <para>
/// The authors' repository is no longer published, so the unstated choices follow the paper's backbone and the ControlNet
/// convention it builds on; each is documented on <see cref="RealisVSROptions"/>. The branch's blocks start as copies of the
/// main network's first blocks and its patch embedding is new. Clips are padded to 4k + 1 frames and to sides the VAE and
/// patch grid divide, and the output is cropped back, so every input frame and pixel has an output.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RealisVSR: Detail-enhanced Diffusion for Real-World 4K Video Super-Resolution",
    "https://arxiv.org/abs/2507.19138",
    Year = 2025,
    Authors = "Weisong Zhao, Jingkai Zhou, Xiangyu Zhu, Weihua Chen, Xiao-Yu Zhang, Zhen Lei, Fan Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 1e-4,
                Source = "Zhao et al. 2025, Sec. 4.1: AdamW with a learning rate of 1e-5 and a weight decay of 1e-4.")]
public partial class RealisVSR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    // The timesteps the reported objective averages over, a fixed quadrature so it is a deterministic function of the weights.
    private const int ObjectiveTimesteps = 8;

    private readonly RealisVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private WanDiT<T>? _dit;
    private RealisCpc<T>? _cpc;
    private int _promptEmbedding;
    private WanVideoVae<T>? _vae;
    private double[] _sigmas = Array.Empty<double>();
    private Random _random;

    /// <summary>Creates a RealisVSR model in ONNX inference mode.</summary>
    public RealisVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, RealisVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new RealisVSROptions() : new RealisVSROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a RealisVSR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's and Wan 2.1-1.3B's defaults when null.</param>
    /// <param name="optimizer">The ControlNet branch's optimizer; the paper's AdamW when null.</param>
    public RealisVSR(NeuralNetworkArchitecture<T> architecture, RealisVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new RealisVSROptions() : new RealisVSROptions(options);
        Options = _options;
        var o = _options;
        if (o.ModelDim < 1 || o.NumHeads < 1 || o.ModelDim % o.NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ModelDim must split into NumHeads heads.");
        int headDim = o.ModelDim / o.NumHeads;
        if (headDim % 2 != 0 || (headDim / 3) % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The head width must be even with an even third for the 3D rotary embedding.");
        if (o.NumLayers < 1 || o.FfnDim < 1 || o.FrequencyDim < 2 || o.TextDim < 1 || o.TextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The DiT's sizes must be positive.");
        if (o.ControlNetLayers < 1 || o.ControlNetLayers > o.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(options), "The ControlNet branch needs between one and NumLayers blocks.");
        if (o.VaeDimMultipliers is not { Length: 4 } || o.VaeDimMultipliers.Any(m => m < 1) || o.VaeDim < 1 || o.VaeResidualBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Wan's VAE needs four stages (x8 in space, x4 in time).");
        if (o.WaveletWeights is not { Length: 4 } || o.HogBins < 1 || o.HogCellSize < 1 || o.HogBlockSize < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The loss needs four wavelet weights and positive HOG sizes.");
        if (o.ScaleFactor < 1 || o.SamplingSteps < 1 || o.SigmaShift <= 0 || o.NumTrainTimesteps < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor, the sampler and the schedule must be valid.");
        _useNativeMode = true;
        ScaleFactor = o.ScaleFactor;
        NumFrames = o.NumFrames;
        _random = o.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        // Wan's VAE only supplies the latents of the clips and of their condition; it is never trained.
        RegisterParameterComponent("RealisVSR.Vae", _vae, AiDotNet.Models.Parameters.ParameterSlotRole.Frozen);
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
        // The main network has no LoRA adapters: it is held frozen and only the branch trains.
        _dit = new WanDiT<T>(Layers, WanVideoVae<T>.LatentChannels, o.ModelDim, o.FfnDim, o.NumHeads, o.NumLayers, o.FrequencyDim,
            0, o.NormEpsilon);
        _promptEmbedding = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.TextLength * o.TextDim, 0.0));
        _cpc = new RealisCpc<T>(Layers, o.ModelDim, o.NumHeads, o.FfnDim, o.ControlNetLayers, o.NormEpsilon, o.ConditionScale);
        _vae = new WanVideoVae<T>(o.VaeDim, o.VaeDimMultipliers, o.VaeResidualBlocks);
        _sigmas = FlashVSR<T>.Sigmas(o.NumTrainTimesteps, o.SigmaShift);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using (new NoGradScope<T>())
        {
            // One pass on a single-token latent sizes every linear map of both networks.
            var latent = new Tensor<T>(new[] { 1, WanVideoVae<T>.LatentChannels, 1, 2, 2 });
            Velocity(latent, new[] { 1000.0 }, latent);
        }
        CopyMainBlocksIntoBranch();
        MarkLayerShapesResolved();
    }

    // The ControlNet initialization: branch block j starts as main block j.
    private void CopyMainBlocksIntoBranch()
    {
        var main = Dit.Blocks;
        var branch = Cpc.Blocks;
        for (int j = 0; j < branch.Count; j++)
            for (int k = 0; k < main[j].EndLayer - main[j].FirstLayer; k++)
                Layers[branch[j].FirstLayer + k].SetParameters(Layers[main[j].FirstLayer + k].GetParameters());
    }

    private WanDiT<T> Dit => _dit ?? throw new InvalidOperationException("RealisVSR's native layers are not initialized.");

    private RealisCpc<T> Cpc => _cpc ?? throw new InvalidOperationException("RealisVSR's native layers are not initialized.");

    private WanVideoVae<T> Vae => _vae ?? throw new InvalidOperationException("RealisVSR's native layers are not initialized.");

    /// <summary>The ControlNet branch's weights, the only ones RealisVSR trains.</summary>
    internal IReadOnlyList<Tensor<T>> ControlNetParameters => _usingCustomLayers || _cpc is null
        ? Array.Empty<Tensor<T>>()
        : LayerParameters(_cpc.FirstLayer, _cpc.EndLayer);

    /// <summary>The frozen Wan DiT's weights and its prompt embedding.</summary>
    internal IReadOnlyList<Tensor<T>> BackboneParameters => _usingCustomLayers || _cpc is null
        ? Array.Empty<Tensor<T>>()
        : LayerParameters(0, _cpc.FirstLayer);

    /// <summary>The weights of the branch's blocks, in block order.</summary>
    internal IReadOnlyList<Tensor<T>> BranchBlockParameters => _usingCustomLayers || _cpc is null
        ? Array.Empty<Tensor<T>>()
        : _cpc.Blocks.SelectMany(block => LayerParameters(block.FirstLayer, block.EndLayer)).ToList();

    /// <summary>The weights of as many of the main network's first blocks as the branch has, in block order.</summary>
    internal IReadOnlyList<Tensor<T>> LeadingMainBlockParameters => _usingCustomLayers || _cpc is null || _dit is null
        ? Array.Empty<Tensor<T>>()
        : _dit.Blocks.Take(_cpc.Blocks.Count).SelectMany(block => LayerParameters(block.FirstLayer, block.EndLayer)).ToList();

    private IReadOnlyList<Tensor<T>> LayerParameters(int first, int end)
        => TapeTrainingStep<T>.CollectParameters(Enumerable.Range(first, end - first).Select(i => Layers[i]));

    #endregion

    #region Pipeline

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

    // The smallest 4k + 1 frames covering the clip (the VAE's causal layout).
    private static int PaddedLength(int frames) => (frames - 1 + 3) / 4 * 4 + 1;

    // Repeats the last frame of [B, F, ...] until there are `length` frames.
    private Tensor<T> RepeatLast(Tensor<T> clips, int length)
    {
        int f = clips.Shape[1];
        if (f >= length) return clips;
        var target = clips.Shape.ToArray();
        target[1] = length - f;
        return Engine.TensorConcatenate(new[] { clips, Engine.TensorBroadcastTo(Engine.TensorNarrow(clips, 1, f - 1, 1), target) }, 1);
    }

    // The LR side unit: s times it is a multiple of the VAE's x8 times the DiT's x2 patch.
    private int LowUnit
    {
        get
        {
            int unit = 2 * Vae.SpatialFactor, a = unit, b = _options.ScaleFactor;
            while (b != 0) (a, b) = (b, a % b);
            return unit / a;
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

    // LR clips [B, F, 3, h, w] in [0, 1] to the bicubic upscale [B, 3, F, s·h, s·w] in [-1, 1].
    private Tensor<T> Upsample(Tensor<T> clips)
    {
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var up = Engine.Interpolate(Engine.Reshape(clips, new[] { b * f, 3, h, w }), new[] { h * s, w * s }, InterpolateMode.Bicubic, false);
        return Engine.TensorPermute(Engine.Reshape(ToSigned(up), new[] { b, f, 3, h * s, w * s }), new[] { 0, 2, 1, 3, 4 });
    }

    /// <summary>
    /// Eq. 1: the main network's velocity for latents [B, 16, n, h, w] at per-sample timesteps, with the branch's feature
    /// ⌊i / r⌋ (r = ⌈L_main / L_CPC⌉) added after main block i.
    /// </summary>
    private Tensor<T> Velocity(Tensor<T> latents, IReadOnlyList<double> timesteps, Tensor<T> condition)
    {
        var dit = Dit;
        int f = latents.Shape[2], h = latents.Shape[3] / 2, w = latents.Shape[4] / 2;
        var x = dit.Tokens(Engine, Layers, latents);
        var c = dit.Condition(Engine, Layers, timesteps, Prompt(latents.Shape[0]), f, h, w);
        var features = Cpc.Forward(Engine, Layers, condition, c);
        var blocks = dit.Blocks;
        int r = (blocks.Count + features.Count - 1) / features.Count;
        for (int i = 0; i < blocks.Count; i++)
        {
            x = blocks[i].Forward(Engine, Layers, x, c.Context, c.TimeModulation, 0, null, c.Cos, c.Sin);
            if (i / r < features.Count) x = Engine.TensorAdd(x, features[i / r]);
        }
        return dit.Head(Engine, Layers, x, c.Time, f, h, w);
    }

    // Euler on the rectified flow x_σ = (1 − σ)·z + σ·ε from σ = 1 to 0 on Wan's shifted grid: x += (σ' − σ)·v(x, 1000σ).
    private Tensor<T> Sample(Tensor<T> condition, Random random)
    {
        int b = condition.Shape[0], n = _options.SamplingSteps;
        double shift = _options.SigmaShift;
        double Shifted(double sigma) => shift * sigma / (1.0 + (shift - 1.0) * sigma);
        var x = Gaussian(condition.Shape.ToArray(), random);
        for (int i = 0; i < n; i++)
        {
            double sigma = Shifted(1.0 - i / (double)n), next = Shifted(1.0 - (i + 1) / (double)n);
            var v = Velocity(x, Enumerable.Repeat(1000.0 * sigma, b).ToArray(), condition);
            x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(v, NumOps.FromDouble(next - sigma)));
        }
        return x;
    }

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the output size, in [0, 1].
    private Tensor<T> RunClips(Tensor<T> frames, Random random)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int b = clips.Shape[0], f = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], s = _options.ScaleFactor;
        var lq = Upsample(PadSides(RepeatLast(clips, PaddedLength(f)), LowUnit));
        var video = Vae.Decode(Sample(Vae.Encode(lq), random));
        video = Engine.TensorSlice(video, new[] { 0, 0, 0, 0, 0 }, new[] { b, 3, f, h * s, w * s });
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

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ControlNetOptimizer()
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
            }));

    private Tensor<T> Mse(Tensor<T> a, Tensor<T> b)
    {
        var squared = Engine.TensorSquare(Engine.TensorSubtract(a, b));
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), false);
    }

    /// <summary>One training clip: the ground truth's latents and the condition's.</summary>
    private sealed class Batch
    {
        internal Batch(Tensor<T> latents, Tensor<T> condition)
        {
            Latents = latents;
            Condition = condition;
        }

        /// <summary>The ground truth's Wan VAE latents [B, 16, 1 + k, H / 8, W / 8].</summary>
        internal Tensor<T> Latents { get; }

        /// <summary>The bicubic upscale's latents, the same shape.</summary>
        internal Tensor<T> Condition { get; }
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
        var truth = PadSides(RepeatLast(high, length), LowUnit * _options.ScaleFactor);
        return new Batch(Vae.Encode(Engine.TensorPermute(ToSigned(truth), new[] { 0, 2, 1, 3, 4 })), Vae.Encode(lq));
    }

    /// <summary>
    /// The High-Frequency Rectified Diffusion Loss at per-sample σ: on the velocity v against v* = ε − z for
    /// x_σ = (1 − σ)·z + σ·ε, the MSE plus the wavelet loss (Eq. 4) plus the HOG loss (Eq. 5).
    /// </summary>
    private Tensor<T> DiffusionLoss(Batch batch, IReadOnlyList<double> sigmas, Tensor<T> noise)
    {
        var latents = batch.Latents;
        var shape = latents.Shape.ToArray();
        var column = new Tensor<T>(new[] { sigmas.Count, 1, 1, 1, 1 }, new Vector<T>(sigmas.Select(s => NumOps.FromDouble(s)).ToArray()));
        var sigma = Engine.TensorBroadcastTo(column, shape);
        var noisy = Engine.TensorAdd(Engine.TensorMultiply(latents, Engine.TensorAddScalar(Engine.TensorNegate(sigma), NumOps.One)),
            Engine.TensorMultiply(noise, sigma));
        var target = Engine.TensorSubtract(noise, latents);
        var v = Velocity(noisy, sigmas.Select(s => 1000.0 * s).ToArray(), batch.Condition);
        var o = _options;
        var loss = Engine.TensorAdd(Mse(v, target), RealisLoss.Wavelet(Engine, v, target, o.WaveletWeights));
        return Engine.TensorAdd(loss, RealisLoss.Hog(Engine, v, target, o.HogBins, o.HogCellSize, o.HogBlockSize));
    }

    // The loss at uniformly drawn training σ with fresh noise.
    private Tensor<T> DiffusionLoss(Batch batch, Random random)
    {
        var sigmas = Enumerable.Range(0, batch.Latents.Shape[0]).Select(_ => _sigmas[random.Next(_sigmas.Length)]).ToArray();
        return DiffusionLoss(batch, sigmas, Gaussian(batch.Latents.Shape.ToArray(), random));
    }

    // The loss averaged over the schedule's quantile midpoints with seeded noise: what the drawn loss estimates.
    private Tensor<T> DiffusionObjective(Batch batch)
    {
        var random = RandomHelper.CreateSeededRandom(1729);
        int b = batch.Latents.Shape[0], n = _sigmas.Length;
        Tensor<T>? total = null;
        for (int q = 0; q < ObjectiveTimesteps; q++)
        {
            double sigma = _sigmas[Math.Min(n - 1, (int)((q + 0.5) * n / ObjectiveTimesteps))];
            var term = DiffusionLoss(batch, Enumerable.Repeat(sigma, b).ToArray(), Gaussian(batch.Latents.Shape.ToArray(), random));
            total = total is null ? term : Engine.TensorAdd(total, term);
        }
        return Engine.TensorMultiplyScalar(total ?? new Tensor<T>(new[] { 1 }), NumOps.FromDouble(1.0 / ObjectiveTimesteps));
    }

    /// <inheritdoc />
    /// <remarks>The ControlNet branch alone, on the High-Frequency Rectified Diffusion Loss; Wan and its VAE stay frozen.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var batch = Prepare(input, expectedOutput);
        var random = RandomHelper.CreateSeededRandom(_random.Next());
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, _ => DiffusionLoss(batch, random))
            {
                Optimizer = ControlNetOptimizer(),
                Parameters = () => ControlNetParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.DiffusionDenoising;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        return DiffusionObjective(Prepare(input, target))[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? ControlNetOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    // Each step draws its σ and noise on the host, which a compiled plan would freeze at its first step.
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
            Name = _useNativeMode ? "RealisVSR-Native" : "RealisVSR-ONNX",
            Description = "RealisVSR detail-enhanced video super-resolution on Wan 2.1 (Zhao et al. 2025)",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumLayers"] = _options.NumLayers.ToString();
        m.AdditionalInfo["ControlNetLayers"] = _options.ControlNetLayers.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealisVSR<T>));
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
