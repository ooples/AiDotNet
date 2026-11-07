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
/// SeedVR: seeding infinity in diffusion transformer towards generic video restoration, as the authors released it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SeedVR (Wang et al. 2025) restores video with a rectified-flow MM-DiT whose attention runs in large windows: each block
/// cuts the latent token grid into windows (a fixed count at 720p, so any resolution works), every window's video tokens
/// attend to themselves and the prompt with a window-local 3D rotary embedding, and alternate blocks shift the windows by half.
/// Video and text keep separate weights throughout. The low-quality clip enters as its causal-VAE latent, lightly noised and
/// concatenated channel-wise with the noisy latent and a mask channel.
/// </para>
/// <para>
/// Sampling follows the release: 50 Euler steps over the uniform trailing timesteps with velocity (v = ε − x₀) prediction,
/// classifier-free guidance between the precomputed positive and negative prompt embeddings, and the clean estimate at the
/// last step. Training is SD3's flow matching, which the paper follows: logit-normal timesteps shifted with the resolution
/// (the release's <c>timestep_transform</c>), the MSE on the velocity, the condition noised at a quarter of the schedule.
/// </para>
/// <para>
/// <b>For Beginners:</b> SeedVR fixes degraded video (blur, noise, compression, low resolution) with a large generative
/// model. It looks at the video in big overlapping tiles so it can handle any size, and uses a learned picture of what clean
/// video looks like to fill in lost detail.
/// </para>
/// <para>
/// The prompt embeddings are part of the release (text-encoder outputs computed once) and are held frozen; untrained, they
/// are zero. The input is upscaled bicubically by <see cref="SeedVROptions.ScaleFactor"/> in place of the release's resize to
/// the 720p area, clips are padded to 4k + 1 frames and to multiples of 16 pixels, and the output is cropped back.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration",
    "https://arxiv.org/abs/2501.01320",
    Year = 2025,
    Authors = "Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Fei Xiao, Chen Change Loy, Lu Jiang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.0,
                Source = "Wang et al. 2025, Sec. 4.1: \"We mostly follow the training settings in SD3\"; SD3 (Esser et al. 2024) "
                        + "trains with AdamW at a learning rate of 1e-4.")]
public partial class SeedVR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    // The timesteps the reported objective averages over: the logit-normal's quantile midpoints.
    private const int ObjectiveTimesteps = 8;
    private const double ScheduleLength = 1000.0;

    private readonly SeedVROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private SeedVrDiT<T>? _dit;
    private int _positivePrompt, _negativePrompt;
    private SeedVrVideoVae<T>? _vae;
    private Random _random;

    /// <summary>Creates a SeedVR model in ONNX inference mode.</summary>
    public SeedVR(NeuralNetworkArchitecture<T> architecture, string modelPath, SeedVROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new SeedVROptions() : new SeedVROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _random = RandomHelper.CreateSecureRandom();
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a SeedVR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the released SeedVR-7B when null.</param>
    /// <param name="optimizer">The DiT's optimizer; SD3's AdamW when null.</param>
    public SeedVR(NeuralNetworkArchitecture<T> architecture, SeedVROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new SeedVROptions() : new SeedVROptions(options);
        Options = _options;
        var o = _options;
        if (o.ModelDim < 1 || o.NumHeads < 1 || o.HeadDim < 6 || o.NumLayers < 1 || o.ExpandRatio < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The DiT's sizes must be positive, the head at least 6 wide for the 3D rotary embedding.");
        if (o.TextDim < 1 || o.TextLength < 1 || o.NegativeTextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The prompt embeddings need a width and tokens.");
        if (o.WindowFrames < 1 || o.WindowRows < 1 || o.WindowColumns < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "The window counts must be positive.");
        if (o.VaeChannels is not { Length: 4 } || o.VaeGroups < 1 || o.VaeChannels.Any(c => c < 1 || c % o.VaeGroups != 0)
            || o.VaeLayersPerBlock < 1 || o.LatentChannels < 1 || o.LatentScaling <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The VAE needs four widths its group count divides.");
        if (o.ScaleFactor < 1 || o.SamplingSteps < 1 || o.ConditionNoiseScale < 0 || o.ConditionNoiseScale > 1
            || o.TrainingConditionNoiseScale < 0 || o.TrainingConditionNoiseScale > 1)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor, the sampler and the condition noise must be valid.");
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
        // The VAE supplies the latents of the clips and their conditions; the release never trains it with the DiT.
        RegisterParameterComponent("SeedVR.Vae", _vae, AiDotNet.Models.Parameters.ParameterSlotRole.Frozen);
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
        _dit = new SeedVrDiT<T>(Layers, o.LatentChannels, o.ModelDim, o.NumHeads, o.HeadDim, o.ExpandRatio, o.NumLayers, o.NormEpsilon,
            o.QkBias, (o.WindowFrames, o.WindowRows, o.WindowColumns));
        _positivePrompt = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.TextLength * o.TextDim, 0.0));
        _negativePrompt = Layers.Count;
        Layers.Add(RealViformerBlocks.Vector<T>(o.NegativeTextLength * o.TextDim, 0.0));
        _vae = new SeedVrVideoVae<T>(o.VaeChannels, o.VaeLayersPerBlock, o.VaeGroups, o.LatentChannels, o.LatentScaling);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using (new NoGradScope<T>())
        {
            // One pass on a single-token latent sizes every linear map.
            var latent = new Tensor<T>(new[] { 1, _options.LatentChannels, 1, 2, 2 });
            Velocity(latent, Condition(latent, 0.0, null), new[] { ScheduleLength }, Prompt(_positivePrompt, _options.TextLength, 1));
        }
        MarkLayerShapesResolved();
    }

    private SeedVrDiT<T> Dit => _dit ?? throw new InvalidOperationException("SeedVR's native layers are not initialized.");

    private SeedVrVideoVae<T> Vae => _vae ?? throw new InvalidOperationException("SeedVR's native layers are not initialized.");

    /// <summary>The DiT's weights, the ones SeedVR trains.</summary>
    internal IReadOnlyList<Tensor<T>> DiTParameters => _usingCustomLayers || _dit is null
        ? Array.Empty<Tensor<T>>()
        : TapeTrainingStep<T>.CollectParameters(Enumerable.Range(0, _positivePrompt).Select(i => Layers[i]));

    /// <summary>The frozen prompt embeddings.</summary>
    internal IReadOnlyList<Tensor<T>> PromptParameters => _usingCustomLayers || _dit is null
        ? Array.Empty<Tensor<T>>()
        : TapeTrainingStep<T>.CollectParameters(new[] { Layers[_positivePrompt], Layers[_negativePrompt] });

    #endregion

    #region Schedule

    /// <summary>
    /// The release's <c>timestep_transform</c>: timesteps t in [0, 1000] shifted to s·u / (1 + (s − 1)·u), u = t / 1000, with s
    /// linear in the pixel count: 1 at 256² x 37 frames and 5 at 1280 x 720 x 145 for video, 1 at 256² and 3.2 at 1024² for a
    /// single frame.
    /// </summary>
    /// <param name="timestep">The timestep in [0, 1000].</param>
    /// <param name="latentFrames">The latent frames.</param>
    /// <param name="latentHeight">The latent height.</param>
    /// <param name="latentWidth">The latent width.</param>
    internal static double ShiftTimestep(double timestep, int latentFrames, int latentHeight, int latentWidth)
    {
        double frames = (latentFrames - 1.0) * 4 + 1, pixels = latentHeight * 8.0 * latentWidth * 8.0;
        static double Line(double x, double x1, double y1, double x2, double y2) => y1 + (y2 - y1) / (x2 - x1) * (x - x1);
        double shift = frames > 1
            ? Line(pixels * frames, 256.0 * 256 * 37, 1.0, 1280.0 * 720 * 145, 5.0)
            : Line(pixels, 256.0 * 256, 1.0, 1024.0 * 1024, 3.2);
        double u = timestep / ScheduleLength;
        return ScheduleLength * shift * u / (1.0 + (shift - 1.0) * u);
    }

    /// <summary>The release's uniform trailing sampling timesteps: 1000·(1 − i/n) for i = 0..n − 1.</summary>
    internal static double[] SamplingTimesteps(int steps)
        => Enumerable.Range(0, steps).Select(i => ScheduleLength * (1.0 - i / (double)steps)).ToArray();

    #endregion

    #region Pipeline

    private Tensor<T> Prompt(int layer, int length, int batch)
    {
        var prompt = Engine.Reshape(RealViformerBlocks.Read(Layers, layer), new[] { 1, length, _options.TextDim });
        return Engine.TensorBroadcastTo(prompt, new[] { batch, length, _options.TextDim });
    }

    private Tensor<T> Positive(int batch) => Prompt(_positivePrompt, _options.TextLength, batch);

    private Tensor<T> Negative(int batch) => Prompt(_negativePrompt, _options.NegativeTextLength, batch);

    // Inference draws its noise from a fixed seed so the same clip always restores to the same video.
    private Random InferenceRandom() => RandomHelper.CreateSeededRandom(_options.Seed ?? 666);

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

    private Tensor<T> RepeatLast(Tensor<T> clips, int length)
    {
        int f = clips.Shape[1];
        if (f >= length) return clips;
        var target = clips.Shape.ToArray();
        target[1] = length - f;
        return Engine.TensorConcatenate(new[] { clips, Engine.TensorBroadcastTo(Engine.TensorNarrow(clips, 1, f - 1, 1), target) }, 1);
    }

    // The LR side unit: s times it is a multiple of 16 (the VAE's x8 and the DiT's x2 patch; the release's DivisibleCrop(16)).
    private int LowUnit
    {
        get
        {
            int a = 16, b = _options.ScaleFactor;
            while (b != 0) (a, b) = (b, a % b);
            return 16 / a;
        }
    }

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
        up = Engine.TensorClamp(up, NumOps.Zero, NumOps.One);
        return Engine.TensorPermute(Engine.Reshape(ToSigned(up), new[] { b, f, 3, h * s, w * s }), new[] { 0, 2, 1, 3, 4 });
    }

    /// <summary>
    /// The DiT's input condition (<c>get_condition</c> for the "sr" task): the LQ latent noised to the shifted timestep
    /// 1000·<paramref name="noiseScale"/> on the lerp schedule, then a channel of ones.
    /// </summary>
    private Tensor<T> Condition(Tensor<T> latent, double noiseScale, Random? random)
    {
        var shape = latent.Shape.ToArray();
        var noised = latent;
        if (noiseScale > 0 && random is not null)
        {
            double tau = ShiftTimestep(ScheduleLength * noiseScale, shape[2], shape[3], shape[4]) / ScheduleLength;
            noised = Engine.TensorAdd(Engine.TensorMultiplyScalar(latent, NumOps.FromDouble(1.0 - tau)),
                Engine.TensorMultiplyScalar(Gaussian(shape, random), NumOps.FromDouble(tau)));
        }
        var mask = new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3], shape[4] });
        mask.Fill(NumOps.One);
        return Engine.TensorConcatenate(new[] { noised, mask }, 1);
    }

    // The velocity prediction for x_t [B, C, n, h, w] given the condition [B, C + 1, n, h, w].
    private Tensor<T> Velocity(Tensor<T> latents, Tensor<T> condition, IReadOnlyList<double> timesteps, Tensor<T> prompt)
        => Dit.Forward(Engine, Layers, Engine.TensorConcatenate(new[] { latents, condition }, 1), prompt, timesteps);

    // classifier_free_guidance: neg + s·(pos − neg), optionally rescaled toward pos's standard deviation.
    private Tensor<T> Guided(Tensor<T> latents, Tensor<T> condition, IReadOnlyList<double> timesteps)
    {
        int b = latents.Shape[0];
        double scale = _options.GuidanceScale;
        var positive = Velocity(latents, condition, timesteps, Positive(b));
        if (scale == 1.0) return positive;
        var negative = Velocity(latents, condition, timesteps, Negative(b));
        var guided = Engine.TensorAdd(negative, Engine.TensorMultiplyScalar(Engine.TensorSubtract(positive, negative), NumOps.FromDouble(scale)));
        double rescale = _options.GuidanceRescale;
        if (rescale == 0.0) return guided;
        double Std(Tensor<T> x)
        {
            double mean = 0, square = 0;
            for (int i = 0; i < x.Length; i++) mean += NumOps.ToDouble(x[i]);
            mean /= x.Length;
            for (int i = 0; i < x.Length; i++) square += Math.Pow(NumOps.ToDouble(x[i]) - mean, 2);
            return Math.Sqrt(square / Math.Max(1, x.Length - 1));
        }
        double factor = rescale * Std(positive) / Math.Max(Std(guided), 1e-12) + (1.0 - rescale);
        return Engine.TensorMultiplyScalar(guided, NumOps.FromDouble(factor));
    }

    // The Euler sampler with v_lerp prediction: x_s = x_t + (s − t)/T·v between the trailing timesteps, then x₀ = x_t − t/T·v.
    private Tensor<T> Sample(Tensor<T> condition, Random random)
    {
        var shape = condition.Shape.ToArray();
        shape[1] -= 1;
        int b = shape[0];
        var x = Gaussian(shape, random);
        var timesteps = SamplingTimesteps(_options.SamplingSteps);
        for (int i = 0; i < timesteps.Length; i++)
        {
            double t = timesteps[i], s = i + 1 < timesteps.Length ? timesteps[i + 1] : 0.0;
            var v = Guided(x, condition, Enumerable.Repeat(t, b).ToArray());
            x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(v, NumOps.FromDouble((s - t) / ScheduleLength)));
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
        var condition = Condition(Vae.Encode(lq), _options.ConditionNoiseScale, random);
        var video = Vae.Decode(Sample(condition, random));
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

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiTOptimizer()
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

    /// <summary>One training clip: the ground truth's latents and the LQ upscale's.</summary>
    private sealed class Batch
    {
        internal Batch(Tensor<T> latents, Tensor<T> lq)
        {
            Latents = latents;
            Lq = lq;
        }

        /// <summary>The ground truth's scaled latents [B, C, 1 + k, H / 8, W / 8].</summary>
        internal Tensor<T> Latents { get; }

        /// <summary>The LQ upscale's latents, the same shape.</summary>
        internal Tensor<T> Lq { get; }
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
    /// Flow matching at per-sample timesteps u ∈ (0, 1): x_t = (1 − u)·x₀ + u·ε, the velocity v = ε − x₀ predicted under the
    /// positive prompt with the condition noised at <see cref="SeedVROptions.TrainingConditionNoiseScale"/>, the MSE.
    /// </summary>
    private Tensor<T> FlowMatchingLoss(Batch batch, IReadOnlyList<double> fractions, Tensor<T> noise, Random conditionRandom)
    {
        var latents = batch.Latents;
        var shape = latents.Shape.ToArray();
        var timesteps = fractions.Select(u => ShiftTimestep(ScheduleLength * u, shape[2], shape[3], shape[4])).ToArray();
        var column = new Tensor<T>(new[] { timesteps.Length, 1, 1, 1, 1 },
            new Vector<T>(timesteps.Select(t => NumOps.FromDouble(t / ScheduleLength)).ToArray()));
        var sigma = Engine.TensorBroadcastTo(column, shape);
        var noisy = Engine.TensorAdd(Engine.TensorMultiply(latents, Engine.TensorAddScalar(Engine.TensorNegate(sigma), NumOps.One)),
            Engine.TensorMultiply(noise, sigma));
        var condition = Condition(batch.Lq, _options.TrainingConditionNoiseScale, conditionRandom);
        var v = Velocity(noisy, condition, timesteps, Positive(shape[0]));
        return Mse(v, Engine.TensorSubtract(noise, latents));
    }

    // SD3's logit-normal(0, 1) timestep fraction.
    private static double LogitNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    private Tensor<T> FlowMatchingLoss(Batch batch, Random random)
    {
        var fractions = Enumerable.Range(0, batch.Latents.Shape[0]).Select(_ => LogitNormal(random)).ToArray();
        return FlowMatchingLoss(batch, fractions, Gaussian(batch.Latents.Shape.ToArray(), random), random);
    }

    // The loss averaged over the logit-normal's quantile midpoints with seeded noise: what the drawn loss estimates.
    private Tensor<T> FlowMatchingObjective(Batch batch)
    {
        var random = RandomHelper.CreateSeededRandom(1729);
        int b = batch.Latents.Shape[0];
        Tensor<T>? total = null;
        for (int q = 0; q < ObjectiveTimesteps; q++)
        {
            double z = StatisticsHelper<double>.CalculateInverseNormalCDF((q + 0.5) / ObjectiveTimesteps);
            double u = 1.0 / (1.0 + Math.Exp(-z));
            var term = FlowMatchingLoss(batch, Enumerable.Repeat(u, b).ToArray(), Gaussian(batch.Latents.Shape.ToArray(), random), random);
            total = total is null ? term : Engine.TensorAdd(total, term);
        }
        return Engine.TensorMultiplyScalar(total ?? new Tensor<T>(new[] { 1 }), NumOps.FromDouble(1.0 / ObjectiveTimesteps));
    }

    /// <inheritdoc />
    /// <remarks>The DiT on the flow-matching loss; the VAE and the prompt embeddings stay frozen.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var batch = Prepare(input, expectedOutput);
        var random = RandomHelper.CreateSeededRandom(_random.Next());
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, _ => FlowMatchingLoss(batch, random))
            {
                Optimizer = DiTOptimizer(),
                Parameters = () => DiTParameters,
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
        return FlowMatchingObjective(Prepare(input, target))[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? DiTOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    // Each step draws its timesteps and noise on the host, which a compiled plan would freeze at its first step.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <summary>Frames are pixels on the 0-255 scale; the network divides them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>Restored frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SeedVR-Native" : "SeedVR-ONNX",
            Description = "SeedVR window-attention diffusion transformer for generic video restoration (Wang et al. 2025)",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumLayers"] = _options.NumLayers.ToString();
        m.AdditionalInfo["ModelDim"] = _options.ModelDim.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeedVR<T>));
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
