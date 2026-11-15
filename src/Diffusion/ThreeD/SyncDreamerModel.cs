using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// SyncDreamer (Liu et al. 2023): synchronized multiview diffusion that generates a fixed ring of 3D-consistent views
/// of the object in one input image, then reconstructs it with NeuS.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All N target views (16, at a fixed 30 degree elevation and 22.5 degrees apart in azimuth) are denoised together by
/// one shared noise predictor, the Zero123-initialized latent-diffusion U-Net, which sees the noisy target latent
/// concatenated with the input view's VAE latent and cross-attends to the input view's CLIP embedding projected with
/// the target viewpoint (Zero123's <c>cc_projection</c>). The views are synchronized by a 3D-aware feature attention:
/// at every step each view's noisy latent is encoded and unprojected into a shared 32^3 spatial feature volume, which
/// a 3D U-Net processes; for each target view a view-frustum volume is sampled from it and processed by a second
/// 3D U-Net, and depth-wise attention in the U-Net's middle and decoder blocks lets each pixel attend to the frustum
/// features along its own ray. Sampling is DDIM with 50 steps, eta 1 and classifier-free guidance 2, the
/// unconditional branch zeroing the CLIP context, the input latent and the frustum features.
/// </para>
/// <para>
/// The generated views have known cameras, so the object is reconstructed as the official code does
/// (<c>train_renderer.py</c>): a NeuS field trained with SyncDreamer's renderer and loss on every pixel of the views
/// (<c>renderer/renderer.py</c>), meshed by marching cubes at its zero level over [-0.6, 0.6]^3; point
/// clouds are colored surface points of that mesh. The official trainer supervises NeuS with CarveKit foreground
/// masks; CarveKit is not available here, so masks come from the caller (<see cref="ReconstructMesh"/>) or the views'
/// alpha.
/// </para>
/// <para>
/// <b>Weights.</b> The architecture is the paper's, but meaningful views need the released SyncDreamer checkpoint (the
/// U-Net and VAE through their own loaders, everything else through <see cref="LoadSyncDreamerWeights"/>). Freshly
/// initialized weights produce the right shapes and a working pipeline, not a plausible object.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give SyncDreamer one photo of an object; it imagines 16 photos of the same object from all
/// around it, makes them agree with each other through a shared 3D feature grid, and then fits a 3D shape to them.
/// </para>
/// <para>
/// Reference: Liu et al., "SyncDreamer: Generating Multiview-consistent Images from a Single-view Image", ICLR 2024
/// (arXiv 2309.03453); official code liuyuan-pal/SyncDreamer.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var model = new SyncDreamerModel&lt;float&gt;();          // the paper's configuration
/// var views = model.GenerateViews(image, seed: 7);       // 16 views [1, 3, 256, 256] in [0, 1]
/// var mesh = model.GenerateMesh(image, seed: 7);         // NeuS + marching cubes, vertex colors
/// var cloud = model.GeneratePointCloud(image, seed: 7);  // [1, P, 6] xyz + rgb
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SyncDreamer: Generating Multiview-consistent Images from a Single-view Image", "https://arxiv.org/abs/2309.03453", Year = 2024, Authors = "Liu et al.")]
public partial class SyncDreamerModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
        RegisterParameterComponent(_network);
    }

    #region Fields

    private const int LatentChannelCount = 4;

    private readonly SyncDreamerOptions _options;
    private readonly UNetNoisePredictor<T> _unet;
    private readonly StandardVAE<T> _vae;
    private readonly SyncDreamerNetwork<T> _network;
    private readonly SyncDreamerCameraLayout _cameras;

    #endregion

    #region Constructor

    /// <summary>
    /// Builds SyncDreamer with the given configuration (the official one when null). Weights are allocated lazily and
    /// initialized from <see cref="ModelOptions.Seed"/> when it is set.
    /// </summary>
    /// <param name="options">The configuration; the official SyncDreamer configuration when null.</param>
    /// <param name="unet">The noise predictor, or null for the configured Zero123 U-Net (8 input channels: the noisy
    /// latent and the input view's latent). Pass a U-Net holding Zero123 or SyncDreamer weights here.</param>
    /// <param name="vae">The autoencoder, or null for the Stable Diffusion KL-f8 VAE.</param>
    /// <param name="scheduler">The training scheduler the base diffusion API uses, or null for DDIM on the
    /// configured schedule. SyncDreamer's own sampler is the synchronized DDIM of <see cref="GenerateViews"/>.</param>
    /// <param name="architecture">An optional architecture handed to the default U-Net.</param>
    public SyncDreamerModel(
        SyncDreamerOptions? options = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        INoiseScheduler<T>? scheduler = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            CreateDiffusionOptions(options),
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            (options ?? new SyncDreamerOptions()).PointCount,
            architecture)
    {
        _options = options ?? new SyncDreamerOptions();
        Validate(_options);
        var o = _options;
        LayerInitializationSeedScope.ResetForModelConstruction(architecture?.RandomSeed ?? o.Seed);
        int latentSide = o.ImageSize / o.VaeDownsampleFactor;
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: 2 * LatentChannelCount,
            outputChannels: LatentChannelCount,
            baseChannels: o.UNetBaseChannels,
            channelMultipliers: (int[])o.UNetChannelMultipliers.Clone(),
            numResBlocks: o.UNetNumResBlocks,
            attentionResolutions: (int[])o.UNetAttentionLevels.Clone(),
            contextDim: o.CrossAttentionDim,
            numHeads: o.UNetHeads,
            inputHeight: latentSide,
            seed: o.Seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LatentChannelCount,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            latentScaleFactor: o.LatentScaleFactor,
            seed: o.Seed);
        _network = new SyncDreamerNetwork<T>(o, LatentChannelCount, _unet.FeatureSites(), _unet.ChannelMultipliers.Length);
        _cameras = new SyncDreamerCameraLayout(o);
    }

    private static DiffusionModelOptions<T> CreateDiffusionOptions(SyncDreamerOptions? options)
    {
        var o = options ?? new SyncDreamerOptions();
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = o.TrainTimesteps,
            BetaStart = o.BetaStart,
            BetaEnd = o.BetaEnd,
            BetaSchedule = BetaSchedule.ScaledLinear,
            Seed = o.Seed,
        };
    }

    private static void Validate(SyncDreamerOptions o)
    {
        void Positive(int value, string name)
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(name, $"{name} must be positive.");
        }

        Positive(o.ViewCount, nameof(o.ViewCount));
        Positive(o.ImageSize, nameof(o.ImageSize));
        Positive(o.VaeDownsampleFactor, nameof(o.VaeDownsampleFactor));
        Positive(o.UNetBaseChannels, nameof(o.UNetBaseChannels));
        Positive(o.UNetNumResBlocks, nameof(o.UNetNumResBlocks));
        Positive(o.UNetHeads, nameof(o.UNetHeads));
        Positive(o.CrossAttentionDim, nameof(o.CrossAttentionDim));
        Positive(o.TimeEmbedDim, nameof(o.TimeEmbedDim));
        Positive(o.TargetEncoderWidth, nameof(o.TargetEncoderWidth));
        Positive(o.TargetEncoderOutputDim, nameof(o.TargetEncoderOutputDim));
        Positive(o.NormGroups, nameof(o.NormGroups));
        Positive(o.SpatialVolumeSize, nameof(o.SpatialVolumeSize));
        Positive(o.FrustumVolumeDepth, nameof(o.FrustumVolumeDepth));
        Positive(o.DepthAttentionHeads, nameof(o.DepthAttentionHeads));
        Positive(o.ClipImageSize, nameof(o.ClipImageSize));
        Positive(o.ClipPatchSize, nameof(o.ClipPatchSize));
        Positive(o.ClipWidth, nameof(o.ClipWidth));
        Positive(o.ClipLayers, nameof(o.ClipLayers));
        Positive(o.ClipHeads, nameof(o.ClipHeads));
        Positive(o.SamplingSteps, nameof(o.SamplingSteps));
        Positive(o.TrainTimesteps, nameof(o.TrainTimesteps));
        Positive(o.PointCount, nameof(o.PointCount));
        if (o.ReconstructionIterations < 0)
            throw new ArgumentOutOfRangeException(nameof(o.ReconstructionIterations), "ReconstructionIterations cannot be negative.");
        if (o.ReconstructionWarmUpIterations < 0)
            throw new ArgumentOutOfRangeException(nameof(o.ReconstructionWarmUpIterations), "ReconstructionWarmUpIterations cannot be negative.");
        if (o.MeshResolution < 2)
            throw new ArgumentOutOfRangeException(nameof(o.MeshResolution), "MeshResolution needs at least two samples per axis.");
        if (o.ImageSize % o.VaeDownsampleFactor != 0)
            throw new ArgumentException($"ImageSize {o.ImageSize} must be a multiple of VaeDownsampleFactor {o.VaeDownsampleFactor}.", nameof(o));
        int latentSide = o.ImageSize / o.VaeDownsampleFactor;
        // The 3D U-Nets halve every side three times and double back, so every side must divide by 8.
        foreach (var (side, name) in new[]
                 {
                     (latentSide, "the latent side"), (o.SpatialVolumeSize, nameof(o.SpatialVolumeSize)),
                     (o.FrustumVolumeDepth, nameof(o.FrustumVolumeDepth)),
                 })
        {
            if (side % 8 != 0) throw new ArgumentException($"{name} ({side}) must be a multiple of 8.", nameof(o));
        }

        if (o.UNetChannelMultipliers is null || o.UNetChannelMultipliers.Length == 0 || o.UNetChannelMultipliers.Length > 4)
            throw new ArgumentException("UNetChannelMultipliers needs one to four levels (the frustum volume has four scales).", nameof(o));
        if (o.UNetAttentionLevels is null)
            throw new ArgumentException("UNetAttentionLevels cannot be null.", nameof(o));
        if (o.SpatialVolumeDims is null || o.SpatialVolumeDims.Length != 4)
            throw new ArgumentException("SpatialVolumeDims needs four widths.", nameof(o));
        if (o.FrustumVolumeDims is null || o.FrustumVolumeDims.Length != 4)
            throw new ArgumentException("FrustumVolumeDims needs four widths.", nameof(o));
        foreach (int width in o.FrustumVolumeDims)
            if (width <= 0 || width % 2 != 0)
                throw new ArgumentException("FrustumVolumeDims must be positive and even (the depth heads are d / 2 wide).", nameof(o));
        if (o.TimeEmbedDim % 2 != 0) throw new ArgumentException("TimeEmbedDim must be even.", nameof(o));
        if (!(o.CameraDistance > 0) || double.IsInfinity(o.CameraDistance))
            throw new ArgumentOutOfRangeException(nameof(o.CameraDistance), "CameraDistance must be positive and finite.");
        if (!(o.FocalLength > 0) || double.IsInfinity(o.FocalLength))
            throw new ArgumentOutOfRangeException(nameof(o.FocalLength), "FocalLength must be positive and finite.");
        if (!(o.SpatialVolumeLength > 0) || !(o.FrustumVolumeLength > 0))
            throw new ArgumentOutOfRangeException(nameof(o.SpatialVolumeLength), "The volume extents must be positive.");
        if (!(o.LatentScaleFactor > 0)) throw new ArgumentOutOfRangeException(nameof(o.LatentScaleFactor));
        if (double.IsNaN(o.TargetElevationDegrees) || double.IsNaN(o.InputElevationDegrees))
            throw new ArgumentException("The elevations must be numbers.", nameof(o));
        if (!(o.GuidanceScale >= 0) || !(o.DdimEta >= 0))
            throw new ArgumentOutOfRangeException(nameof(o.GuidanceScale), "GuidanceScale and DdimEta cannot be negative.");
        if (!(o.BetaStart > 0) || !(o.BetaEnd > o.BetaStart) || !(o.BetaEnd < 1))
            throw new ArgumentOutOfRangeException(nameof(o.BetaEnd), "The betas must satisfy 0 < BetaStart < BetaEnd < 1.");
    }

    #endregion

    #region Properties

    /// <summary>The configuration this model was built with.</summary>
    public SyncDreamerOptions Configuration => _options;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <summary>SyncDreamer is conditioned on an image, not on text.</summary>
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => LatentChannelCount;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <summary>The mesh carries per-vertex colors from the NeuS color network, but no texture image is baked.</summary>
    public override bool SupportsTexture => false;

    /// <inheritdoc />
    public override bool SupportsNovelView => true;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => false;

    /// <summary>The number of synchronized views generated together.</summary>
    public int NumViews => _options.ViewCount;

    /// <summary>
    /// The NeuS field configuration (network widths, samples per ray, grid chunking) of the reconstruction, or null for
    /// NeuS's defaults, which are SyncDreamer's (<c>renderer/renderer.py NeuSRenderer</c>: SDF 8 x 256 with a skip at 4,
    /// 6 frequencies, color 4 x 256, 64 + 64 samples, 4 up-sampling steps). The renderer variant, bounds, annealing
    /// and background are always SyncDreamer's.
    /// </summary>
    internal NeuSOptions? ReconstructionOptions { get; set; }

    internal SyncDreamerNetwork<T> Network => _network;

    internal SyncDreamerCameraLayout Cameras => _cameras;

    internal UNetNoisePredictor<T> UNet => _unet;

    #endregion

    #region Multiview generation

    /// <summary>
    /// Generates the synchronized target views of the object in <paramref name="image"/>: view i at azimuth
    /// <c>i * 360 / N</c> degrees from the input view and the configured target elevation.
    /// </summary>
    /// <param name="image">The input view, [1, 3, H, W] or [3, H, W] with values in [0, 1]; resized to
    /// <see cref="SyncDreamerOptions.ImageSize"/> when it differs.</param>
    /// <param name="numInferenceSteps">The DDIM steps, or null for <see cref="SyncDreamerOptions.SamplingSteps"/>.</param>
    /// <param name="guidanceScale">The guidance scale, or null for <see cref="SyncDreamerOptions.GuidanceScale"/>.</param>
    /// <param name="seed">The sampling seed.</param>
    /// <returns>The views, each [1, 3, S, S] with values in [0, 1].</returns>
    public Tensor<T>[] GenerateViews(Tensor<T> image, int? numInferenceSteps = null, double? guidanceScale = null, int? seed = null)
    {
        var latents = SampleLatents(image, numInferenceSteps ?? _options.SamplingSteps, guidanceScale ?? _options.GuidanceScale, seed);
        int n = latents.Shape[0], s = latents.Shape[2];
        var views = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            var latent = Engine.TensorSlice(latents, new[] { i, 0, 0, 0 }, new[] { 1, LatentChannelCount, s, s });
            views[i] = ToUnitRange(DecodeFromLatent(latent));
        }

        return views;
    }

    /// <summary>The denoised target latents [N, 4, S/8, S/8] (scaled as the U-Net sees them).</summary>
    internal Tensor<T> SampleLatents(Tensor<T> image, int steps, double guidanceScale, int? seed)
    {
        var rng = CreateInferenceRng(seed);
        var input = EncodeInputView(image, rng);
        var schedule = SyncDreamerDdimSchedule.Create(_options.TrainTimesteps, _options.BetaStart, _options.BetaEnd, steps, _options.DdimEta);
        int side = _cameras.LatentSize;
        var x = SampleNoiseTensor(new[] { _options.ViewCount, LatentChannelCount, side, side }, rng);
        for (int index = schedule.Count - 1; index >= 0; index--)
            x = DenoiseStep(x, input, schedule, index, guidanceScale, rng);
        return x;
    }

    /// <summary>
    /// One synchronized DDIM step (<c>SyncDDIMSampler.denoise_apply</c>): the noise of every view from the shared U-Net
    /// with the 3D-aware feature attention, then the DDIM update with eta-scaled fresh noise (none at the last step).
    /// </summary>
    internal Tensor<T> DenoiseStep(Tensor<T> x, SyncDreamerInputView<T> input, SyncDreamerDdimSchedule schedule, int index,
        double guidanceScale, Random rng)
    {
        var noise = PredictMultiviewNoise(x, input, schedule.Timesteps[index], guidanceScale);
        var fresh = index == 0 ? null : SampleNoiseTensor(x.Shape.ToArray(), rng);
        return schedule.Step(Engine, x, noise, index, fresh);
    }

    /// <summary>
    /// The input view's conditioning: its CLIP embedding [1, dim] and its unscaled VAE latent [1, 4, S/8, S/8], a draw
    /// from the VAE posterior (<c>prepare</c> calls <c>encode_first_stage(image_input)</c> with its default
    /// <c>sample=True</c>; <c>DiagonalGaussianDistribution</c> clamps the log-variance to [-30, 20]).
    /// </summary>
    /// <param name="image">The input view.</param>
    /// <param name="rng">The generator the posterior draw takes its seed from.</param>
    internal SyncDreamerInputView<T> EncodeInputView(Tensor<T> image, Random rng)
    {
        if (rng is null) throw new ArgumentNullException(nameof(rng));
        var prepared = PrepareImage(image);
        var clip = _network.Clip.Encode(prepared);
        // The VAE reads [-1, 1]. Zero123's U-Net was trained on the unscaled latent: SyncDreamer multiplies by the
        // scale factor when encoding and divides it back out before the U-Net (UNetWrapper with use_zero_123).
        var (mean, logVariance) = _vae.EncodeWithDistribution(
            Engine.TensorSubtractScalar(Engine.TensorMultiplyScalar(prepared, NumOps.FromDouble(2.0)), NumOps.One));
        logVariance = Engine.TensorClamp(logVariance, NumOps.FromDouble(-30.0), NumOps.FromDouble(20.0));
        var latent = _vae.Sample(mean, logVariance, rng.Next());
        if (latent.Shape[1] != LatentChannelCount || latent.Shape[2] != _cameras.LatentSize || latent.Shape[3] != _cameras.LatentSize)
            throw new InvalidOperationException(
                $"The VAE encodes a {_options.ImageSize}-pixel image to [{string.Join(", ", latent.Shape.ToArray())}], " +
                $"but the layout expects [1, {LatentChannelCount}, {_cameras.LatentSize}, {_cameras.LatentSize}].");
        return new SyncDreamerInputView<T>(clip, latent);
    }

    /// <summary>
    /// The guided noise of every view, [N, 4, S, S]: the spatial volume from all views' noisy latents, then for each
    /// view its frustum features, its viewpoint-projected CLIP context and the U-Net with depth-wise attention.
    /// </summary>
    internal Tensor<T> PredictMultiviewNoise(Tensor<T> x, SyncDreamerInputView<T> input, int timestep, double guidanceScale)
    {
        if (_unet.InputChannels != 2 * LatentChannelCount)
            throw new InvalidOperationException(
                $"SyncDreamer's U-Net reads the noisy and the input latent ({2 * LatentChannelCount} channels); this one reads {_unet.InputChannels}.");
        int n = _options.ViewCount, side = _cameras.LatentSize;
        var geometry = Geometry();
        var timeEmbedding = _network.EmbedTime(timestep);
        var spatial = _network.ConstructSpatialVolume(x, timeEmbedding, geometry.Viewpoints, geometry.Warps);
        bool guided = guidanceScale != 1.0;
        var noises = new Tensor<T>[n];
        for (int view = 0; view < n; view++)
        {
            var viewpoint = Engine.TensorSlice(geometry.Viewpoints, new[] { view, 0 }, new[] { 1, SyncDreamerNetwork<T>.ViewpointDim });
            var frustumGrid = Engine.TensorSlice(geometry.Frustums, new[] { view, 0, 0, 0, 0 },
                new[] { 1, geometry.Frustums.Shape[1], side, side, 3 });
            var frustum = _network.ConstructFrustumVolume(spatial, timeEmbedding, viewpoint, frustumGrid);
            var context = _network.ViewContext(input.Clip, viewpoint);
            var noisy = Engine.TensorSlice(x, new[] { view, 0, 0, 0 }, new[] { 1, LatentChannelCount, side, side });
            var conditional = PredictViewNoise(noisy, input.Latent, timestep, context, frustum);
            if (guided)
            {
                var zeroFrustum = new Tensor<T>[frustum.Length];
                for (int i = 0; i < frustum.Length; i++) zeroFrustum[i] = new Tensor<T>(frustum[i].Shape.ToArray());
                var unconditional = PredictViewNoise(noisy, new Tensor<T>(input.Latent.Shape.ToArray()), timestep,
                    new Tensor<T>(context.Shape.ToArray()), zeroFrustum);
                conditional = ApplyGuidance(unconditional, conditional, guidanceScale);
            }

            noises[view] = conditional;
        }

        return Engine.TensorConcatenate(noises, 0);
    }

    /// <summary>One view's U-Net noise: input [noisy, input latent], CLIP context, depth-wise attention to the frustum.</summary>
    internal Tensor<T> PredictViewNoise(Tensor<T> noisy, Tensor<T> inputLatent, int timestep, Tensor<T> context, Tensor<T>[] frustum)
    {
        var joined = Engine.TensorConcatenate(new[] { noisy, inputLatent }, 1);
        return _unet.PredictNoiseWithFeatureHook(joined, timestep, context, (site, features) =>
            _network.Condition(site.DecoderBlock + 1, site, features, frustum));
    }

    private SyncDreamerGeometry<T>? _geometry;

    // The camera geometry depends on the configuration only; it is built once.
    internal SyncDreamerGeometry<T> Geometry()
    {
        if (_geometry is not null) return _geometry;
        int n = _options.ViewCount, v = _options.SpatialVolumeSize, d = _options.FrustumVolumeDepth, s = _cameras.LatentSize;
        var warps = new double[n][];
        var frustums = new double[n][];
        for (int i = 0; i < n; i++)
        {
            warps[i] = _cameras.SpatialVolumeWarp(i, s);
            frustums[i] = _cameras.FrustumGrid(i);
        }

        _geometry = new SyncDreamerGeometry<T>(
            ToTensor(_cameras.ViewpointEmbedding(), new[] { n, SyncDreamerNetwork<T>.ViewpointDim }),
            ToTensor(Concatenate(warps), new[] { n, v, v * v, 2 }),
            ToTensor(Concatenate(frustums), new[] { n, d, s, s, 3 }));
        return _geometry;
    }

    private static Tensor<T> ToTensor(double[] values, int[] shape) => new Tensor<double>(values, shape).Cast<T>();

    private static double[] Concatenate(double[][] parts)
    {
        int total = 0;
        foreach (var part in parts) total += part.Length;
        var result = new double[total];
        int offset = 0;
        foreach (var part in parts)
        {
            Array.Copy(part, 0, result, offset, part.Length);
            offset += part.Length;
        }

        return result;
    }

    // [1, 3, S, S] in [0, 1] at the configured side: prepare_inputs, Image.resize((S, S), BICUBIC) and then
    // rgb * alpha + 1 - alpha. PIL resizes RGBA premultiplied (RGBA -> RGBa -> resize -> RGBA) and clips each channel to
    // [0, 255], so the composite of its output is resize(rgb * alpha) + 1 - resize(alpha), each clipped to [0, 1].
    internal Tensor<T> PrepareImage(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var batched = image.Rank == 3 ? Engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] }) : image;
        if (batched.Rank != 4 || batched.Shape[0] != 1 || (batched.Shape[1] != 3 && batched.Shape[1] != 4))
            throw new ArgumentException(
                $"SyncDreamer reads one RGB or RGBA image [1, 3 or 4, H, W] or [3 or 4, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].",
                nameof(image));
        int h = batched.Shape[2], w = batched.Shape[3], side = _options.ImageSize;
        bool resize = h != side || w != side;
        if (batched.Shape[1] == 3)
            return resize ? PilBicubic(batched, side) : batched;

        var rgb = Engine.TensorSlice(batched, new[] { 0, 0, 0, 0 }, new[] { 1, 3, h, w });
        var alpha = Engine.TensorSlice(batched, new[] { 0, 3, 0, 0 }, new[] { 1, 1, h, w });
        var premultiplied = Engine.TensorMultiply(rgb, Engine.TensorBroadcastTo(alpha, new[] { 1, 3, h, w }));
        if (resize)
        {
            var resized = PilBicubic(Engine.TensorConcatenate(new[] { premultiplied, alpha }, 1), side);
            premultiplied = Engine.TensorSlice(resized, new[] { 0, 0, 0, 0 }, new[] { 1, 3, side, side });
            alpha = Engine.TensorSlice(resized, new[] { 0, 3, 0, 0 }, new[] { 1, 1, side, side });
        }

        return Engine.TensorAdd(premultiplied,
            Engine.TensorBroadcastTo(Engine.TensorAddScalar(Engine.TensorNegate(alpha), NumOps.One), new[] { 1, 3, side, side }));
    }

    // PIL BICUBIC (a = -0.5, antialiased when shrinking), clipped to [0, 1] as PIL clips to the 8-bit range.
    private Tensor<T> PilBicubic(Tensor<T> images, int side)
        => Engine.TensorClamp(AntialiasedResize<T>.Resize(images, side, side, ResizeFilter.Bicubic), NumOps.Zero, NumOps.One);

    // [-1, 1] decoder output to [0, 1].
    private Tensor<T> ToUnitRange(Tensor<T> decoded)
        => Engine.TensorClamp(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(decoded, NumOps.One), NumOps.FromDouble(0.5)),
            NumOps.Zero, NumOps.One);

    #endregion

    #region Reconstruction

    /// <summary>
    /// Reconstructs the object from the views as SyncDreamer does (<c>train_renderer.py</c>): a NeuS field trained on
    /// every pixel of every view with SyncDreamer's renderer and loss (<c>renderer/renderer.py</c>), then
    /// marching cubes at its zero level over the cube of half side
    /// <see cref="SyncDreamerOptions.ReconstructionBoundHalfSide"/> with the field set to 1 outside the unit sphere
    /// (<c>extract_fields</c>), and vertex colors from the color network seen along the normal
    /// (<c>get_vertex_colors</c>), clamped to [0, 1].
    /// </summary>
    /// <param name="views">The N views in layout order, each [1, 3 or 4, S, S] or [3 or 4, S, S] in [0, 1]; a fourth
    /// channel is the view's foreground alpha.</param>
    /// <param name="masks">Per-view foreground masks [1, 1, S, S], [1, S, S] or [S, S] in [0, 1], or null. The official
    /// trainer predicts them with CarveKit, which is not available here; without masks (and without alpha) the mask term
    /// is off.</param>
    /// <param name="seed">The NeuS seed, or null for <see cref="SyncDreamerOptions.ReconstructionSeed"/>.</param>
    /// <returns>The mesh: vertices [V, 3], faces [F, 3], unit normals and colors [V, 3].</returns>
    public Mesh3D<T> ReconstructMesh(Tensor<T>[] views, Tensor<T>[]? masks = null, int? seed = null)
        => Reconstruct(views, masks, seed, quantize: false);

    // quantize: the views went through generate.py's uint8 PNG, (x * 255).astype(np.uint8) / 255.
    internal Mesh3D<T> Reconstruct(Tensor<T>[] views, Tensor<T>[]? masks, int? seed, bool quantize)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        int n = _options.ViewCount, size = _options.ImageSize, pixels = size * size;
        if (views.Length != n)
            throw new ArgumentException($"SyncDreamer reconstructs from its {n} layout views; got {views.Length}.", nameof(views));
        if (masks is not null && masks.Length != n)
            throw new ArgumentException($"Expected {n} masks; got {masks.Length}.", nameof(masks));
        var cameras = new NeuSCamera[n];
        var colors = new double[n][];
        var alphas = new double[n][];
        bool allAlpha = true;
        for (int i = 0; i < n; i++)
        {
            var view = views[i] ?? throw new ArgumentException($"View {i} is null.", nameof(views));
            var chw = view.Rank == 4 && view.Shape[0] == 1
                ? Engine.Reshape(view, new[] { view.Shape[1], view.Shape[2], view.Shape[3] })
                : view;
            if (chw.Rank != 3 || (chw.Shape[0] != 3 && chw.Shape[0] != 4) || chw.Shape[1] != size || chw.Shape[2] != size)
                throw new ArgumentException(
                    $"View {i} must be [1, 3 or 4, {size}, {size}]; got [{string.Join(", ", view.Shape.ToArray())}].", nameof(views));
            int channels = chw.Shape[0];
            var hwc = NeuSHost<T>.ToDoubles(Engine.TensorPermute(chw, new[] { 1, 2, 0 }));
            var rgb = new double[pixels * 3];
            var alpha = channels == 4 ? new double[pixels] : null;
            for (int p = 0; p < pixels; p++)
            {
                for (int c = 0; c < 3; c++)
                {
                    double value = Math.Min(1.0, Math.Max(0.0, hwc[p * channels + c]));
                    rgb[p * 3 + c] = quantize ? Math.Floor(value * 255.0) / 255.0 : value;
                }

                if (alpha is not null) alpha[p] = hwc[p * channels + 3];
            }

            colors[i] = rgb;
            if (alpha is null) allAlpha = false;
            else alphas[i] = alpha;
            cameras[i] = _cameras.CameraForNeuS(i);
        }

        double[][]? hostMasks = null;
        if (masks is not null)
        {
            hostMasks = new double[n][];
            for (int i = 0; i < n; i++)
            {
                var mask = masks[i] ?? throw new ArgumentException($"Mask {i} is null.", nameof(masks));
                if (mask.Length != pixels)
                    throw new ArgumentException($"Mask {i} must have {size} x {size} values; got {mask.Length}.", nameof(masks));
                hostMasks[i] = NeuSHost<T>.ToDoubles(mask);
            }
        }
        else if (allAlpha)
        {
            hostMasks = alphas;
        }

        var fieldOptions = FieldOptions();
        var trainer = new SyncDreamerNeuSTrainer<T>(_options, fieldOptions, seed ?? _options.ReconstructionSeed, Engine);
        var field = trainer.Fit(SyncDreamerNeuSTrainer<T>.BuildRays(cameras, colors, hostMasks, size), _options.ReconstructionIterations);
        return ExtractMesh(field, fieldOptions.GridChunkSize);
    }

    // SyncDreamer's renderer on the configured (or default) NeuS networks.
    private NeuSOptions FieldOptions()
    {
        var options = ReconstructionOptions ?? new NeuSOptions();
        options.RendererVariant = NeuSRendererVariant.SyncDreamer;
        options.NearFarRadius = _options.ReconstructionNearFarRadius;
        options.CubeHalfSide = _options.ReconstructionBoundHalfSide;
        options.AnnealIterations = _options.ReconstructionAnnealIterations;
        options.BackgroundColor = new[] { 1.0, 1.0, 1.0 };   // color + (1 - mask): the white background
        options.Perturb = true;
        return options;
    }

    // extract_mesh: the SDF on a res^3 grid over [-h, h]^3, set to 1 (outside) beyond the unit sphere, marching cubes
    // at 0, and get_vertex_colors.
    internal Mesh3D<T> ExtractMesh(NeuSField<T> field, int chunkSize)
    {
        double h = _options.ReconstructionBoundHalfSide;
        int resolution = _options.MeshResolution;
        var bounds = new SdfGridBounds(-h, -h, -h, h, h, h);
        var grid = NeuSHost<T>.SampleGrid(Engine, points =>
        {
            var sdf = field.QuerySdf(points);
            var host = NeuSHost<T>.ToDoubles(points);
            int count = host.Length / 3;
            var outside = new bool[count];
            for (int i = 0; i < count; i++)
            {
                double x = host[i * 3], y = host[i * 3 + 1], z = host[i * 3 + 2];
                outside[i] = x * x + y * y + z * z >= 1.0;
            }

            return Engine.TensorWhere(outside, NeuSHost<T>.Filled(1.0, count), sdf);
        }, resolution, bounds, chunkSize);
        var mesh = MarchingCubes.Extract(grid, 0.0, IsoSurfaceInterior.BelowIsoLevel,
            bounds.Min[0], bounds.Min[1], bounds.Min[2], bounds.Spacing(resolution, 0));
        if (mesh.Vertices.Shape[0] > 0)
        {
            var normals = field.QueryNormals(mesh.Vertices);
            mesh.Colors = Engine.TensorClamp(field.QueryColor(mesh.Vertices, normals), NumOps.Zero, NumOps.One);
        }

        return mesh;
    }

    /// <inheritdoc />
    /// <remarks>The views must be the layout's views in order (see <see cref="GenerateViews"/>).</remarks>
    protected override Mesh3D<T> ReconstructFromViews(Tensor<T>[] views, (double azimuth, double elevation)[] angles)
    {
        if (angles is null) throw new ArgumentNullException(nameof(angles));
        var layout = GenerateViewAngles(_options.ViewCount);
        if (angles.Length != layout.Length)
            throw new ArgumentException($"SyncDreamer reconstructs from its {layout.Length} layout views.", nameof(angles));
        for (int i = 0; i < angles.Length; i++)
            if (LayoutIndex(angles[i]) != i)
                throw new ArgumentException($"Angle {i} is not layout view {i}.", nameof(angles));
        return Reconstruct(views, null, null, quantize: false);
    }

    #endregion

    #region I3DDiffusionModel

    /// <summary>
    /// Image to 3D: the synchronized views, then NeuS and marching cubes (<see cref="ReconstructMesh"/>).
    /// </summary>
    /// <param name="inputImage">The input view, [1, 3, H, W] or [3, H, W] in [0, 1].</param>
    /// <param name="numViews">Must equal <see cref="SyncDreamerOptions.ViewCount"/>: SyncDreamer generates its fixed
    /// layout, which its 3D-aware attention was trained on.</param>
    /// <param name="numInferenceSteps">The DDIM steps.</param>
    /// <param name="guidanceScale">The guidance scale (the paper uses 2, which the image overload of <c>GenerateMesh</c> applies by default).</param>
    /// <param name="seed">The seed of the sampling and of NeuS.</param>
    public override Mesh3D<T> ImageTo3D(Tensor<T> inputImage, int numViews = 4, int numInferenceSteps = 50,
        double guidanceScale = 3.0, int? seed = null)
    {
        if (numViews != _options.ViewCount)
            throw new ArgumentException(
                $"SyncDreamer generates its fixed layout of {_options.ViewCount} views; got numViews = {numViews}.", nameof(numViews));
        return Reconstruct(GenerateViews(inputImage, numInferenceSteps, guidanceScale, seed), null, seed, quantize: true);
    }

    /// <summary>The image-to-3D mesh with the configured sampler defaults (50 DDIM steps, guidance 2).</summary>
    /// <param name="image">The input view, [1, 3, H, W] or [3, H, W] in [0, 1].</param>
    /// <param name="numInferenceSteps">The DDIM steps, or null for the configured count.</param>
    /// <param name="guidanceScale">The guidance scale, or null for the configured one.</param>
    /// <param name="seed">The seed of the sampling and of NeuS.</param>
    public Mesh3D<T> GenerateMesh(Tensor<T> image, int? numInferenceSteps = null, double? guidanceScale = null, int? seed = null)
        => Reconstruct(GenerateViews(image, numInferenceSteps, guidanceScale, seed), null, seed, quantize: true);

    /// <summary>
    /// A colored point cloud of the reconstructed surface, [1, P, 6] (xyz, rgb): mesh vertices taken at an even stride,
    /// with their NeuS colors.
    /// </summary>
    /// <param name="image">The input view, [1, 3, H, W] or [3, H, W] in [0, 1].</param>
    /// <param name="numPoints">The most points to keep, or null for <see cref="SyncDreamerOptions.PointCount"/>; a
    /// surface with fewer vertices returns all of them.</param>
    /// <param name="numInferenceSteps">The DDIM steps, or null for the configured count.</param>
    /// <param name="guidanceScale">The guidance scale, or null for the configured one.</param>
    /// <param name="seed">The seed of the sampling and of NeuS.</param>
    /// <exception cref="InvalidOperationException">The reconstructed field has no zero level inside the reconstruction bounds.</exception>
    public Tensor<T> GeneratePointCloud(Tensor<T> image, int? numPoints = null, int? numInferenceSteps = null,
        double? guidanceScale = null, int? seed = null)
    {
        var mesh = GenerateMesh(image, numInferenceSteps, guidanceScale, seed);
        return SurfacePoints(mesh, numPoints ?? _options.PointCount);
    }

    /// <summary>Up to <paramref name="count"/> vertices of <paramref name="mesh"/> at an even stride, with colors, [1, P, 6].</summary>
    internal Tensor<T> SurfacePoints(Mesh3D<T> mesh, int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "The point count must be positive.");
        int vertices = mesh.Vertices.Shape[0];
        if (vertices == 0 || mesh.Colors is null)
            throw new InvalidOperationException("The reconstructed NeuS field has no surface inside the reconstruction bounds.");
        int kept = Math.Min(count, vertices);
        var indices = new int[kept];
        for (int i = 0; i < kept; i++) indices[i] = (int)((long)i * vertices / kept);
        var index = new Tensor<int>(indices, new[] { kept });
        var points = Engine.TensorConcatenate(new[]
        {
            Engine.TensorIndexSelect(mesh.Vertices, index, 0),
            Engine.TensorIndexSelect(mesh.Colors, index, 0),
        }, 1);
        return Engine.Reshape(points, new[] { 1, kept, 6 });
    }

    /// <summary>
    /// The views at <paramref name="targetAngles"/>, which must be views of SyncDreamer's fixed layout (azimuth
    /// <c>2 pi i / N</c>, the target elevation, in radians as <see cref="GenerateViewAngles"/> gives them).
    /// </summary>
    /// <exception cref="ArgumentException">An angle is not a layout view.</exception>
    public override Tensor<T>[] SynthesizeNovelViews(Tensor<T> inputImage, (double azimuth, double elevation)[] targetAngles,
        int numInferenceSteps = 50, double guidanceScale = 3.0, int? seed = null)
    {
        if (targetAngles is null) throw new ArgumentNullException(nameof(targetAngles));
        var indices = new int[targetAngles.Length];
        for (int i = 0; i < targetAngles.Length; i++)
        {
            indices[i] = LayoutIndex(targetAngles[i]);
            if (indices[i] < 0)
                throw new ArgumentException(
                    $"Angle {i} (azimuth {targetAngles[i].azimuth}, elevation {targetAngles[i].elevation}) is not one of " +
                    $"SyncDreamer's {_options.ViewCount} layout views.", nameof(targetAngles));
        }

        var views = GenerateViews(inputImage, numInferenceSteps, guidanceScale, seed);
        var selected = new Tensor<T>[indices.Length];
        for (int i = 0; i < indices.Length; i++) selected[i] = views[indices[i]];
        return selected;
    }

    /// <summary>The layout's view angles in radians: azimuth <c>2 pi i / N</c> at the target elevation.</summary>
    /// <param name="numViews">Must equal <see cref="SyncDreamerOptions.ViewCount"/>.</param>
    protected override (double azimuth, double elevation)[] GenerateViewAngles(int numViews)
    {
        if (numViews != _options.ViewCount)
            throw new ArgumentException($"SyncDreamer's layout has {_options.ViewCount} views.", nameof(numViews));
        var angles = new (double azimuth, double elevation)[numViews];
        double elevation = _options.TargetElevationDegrees * Math.PI / 180.0;
        for (int i = 0; i < numViews; i++) angles[i] = (_cameras.AzimuthDegrees[i] * Math.PI / 180.0, elevation);
        return angles;
    }

    // The layout index of an angle in radians, or -1.
    private int LayoutIndex((double azimuth, double elevation) angle)
    {
        const double tolerance = 1e-6;
        double elevation = _options.TargetElevationDegrees * Math.PI / 180.0;
        if (Math.Abs(angle.elevation - elevation) > tolerance) return -1;
        double turn = 2.0 * Math.PI;
        double azimuth = angle.azimuth % turn;
        if (azimuth < 0) azimuth += turn;
        for (int i = 0; i < _options.ViewCount; i++)
        {
            double difference = Math.Abs(azimuth - _cameras.AzimuthDegrees[i] * Math.PI / 180.0);
            if (difference < tolerance || Math.Abs(difference - turn) < tolerance) return i;
        }

        return -1;
    }

    /// <summary>SyncDreamer has no text conditioning.</summary>
    /// <exception cref="NotSupportedException">Always; use the image overload or <see cref="ImageTo3D"/>.</exception>
    public override Tensor<T> GeneratePointCloud(string prompt, string? negativePrompt = null,
        int? numPoints = null, int numInferenceSteps = 50, double guidanceScale = 7.5, int? seed = null)
        => throw new NotSupportedException(TextUnsupportedMessage);

    /// <summary>SyncDreamer has no text conditioning.</summary>
    /// <exception cref="NotSupportedException">Always; use the image overload or <see cref="ImageTo3D"/>.</exception>
    public override Mesh3D<T> GenerateMesh(string prompt, string? negativePrompt = null,
        int resolution = 256, int numInferenceSteps = 50, double guidanceScale = 7.5, int? seed = null)
        => throw new NotSupportedException(TextUnsupportedMessage);

    internal const string TextUnsupportedMessage =
        "SyncDreamer is conditioned on a single input image (Liu et al., 2023) and has no text conditioning; " +
        "use GenerateMesh(image), GeneratePointCloud(image) or ImageTo3D.";

    #endregion

    #region Weights

    /// <summary>
    /// Loads the SyncDreamer-specific weights from a released checkpoint under its state-dict names: the CLIP image
    /// encoder (<c>clip_image_encoder.model.visual.*</c>), <c>cc_projection</c>, <c>time_embed</c>, the spatial volume
    /// network (<c>spatial_volume.*</c>) and the depth-wise attention (<c>model.diffusion_model.middle_conditions</c>
    /// and <c>output_conditions.*</c>). The U-Net and VAE load through their own loaders.
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and expected shape.</param>
    /// <returns>The names read, in order.</returns>
    public IReadOnlyCollection<string> LoadSyncDreamerWeights(Func<string, int[], double[]> read)
        => KlModule<T>.Load(_network.NamedTensors(string.Empty), read);

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _options;
        var metadata = new ModelMetadata<T>
        {
            Name = "SyncDreamer",
            Version = "1.0",
            Description = "SyncDreamer synchronized multiview diffusion with 3D-aware feature attention and NeuS reconstruction",
            FeatureCount = (int)Math.Min(int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "zero123-unet + spatial-volume + frustum depth attention");
        metadata.SetProperty("num_views", o.ViewCount);
        metadata.SetProperty("target_elevation_degrees", o.TargetElevationDegrees);
        metadata.SetProperty("azimuth_step_degrees", 360.0 / o.ViewCount);
        metadata.SetProperty("cross_attention_dim", o.CrossAttentionDim);
        metadata.SetProperty("base_channels", o.UNetBaseChannels);
        metadata.SetProperty("spatial_volume_size", o.SpatialVolumeSize);
        metadata.SetProperty("frustum_volume_depth", o.FrustumVolumeDepth);
        metadata.SetProperty("reconstruction", "NeuS");
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("sampling_steps", o.SamplingSteps);
        metadata.SetProperty("guidance_scale", o.GuidanceScale);
        metadata.SetProperty("default_points", o.PointCount);
        return metadata;
    }

    #endregion
}

/// <summary>The input view's conditioning: its CLIP embedding [1, dim] and its unscaled VAE latent [1, 4, s, s].</summary>
internal sealed class SyncDreamerInputView<T>
{
    public SyncDreamerInputView(Tensor<T> clip, Tensor<T> latent)
    {
        Clip = clip;
        Latent = latent;
    }

    public Tensor<T> Clip { get; }

    public Tensor<T> Latent { get; }
}

/// <summary>The layout geometry as tensors: viewpoint embeddings [N, 4], voxel warps [N, V, V*V, 2], frustum grids [N, D, S, S, 3].</summary>
internal sealed class SyncDreamerGeometry<T>
{
    public SyncDreamerGeometry(Tensor<T> viewpoints, Tensor<T> warps, Tensor<T> frustums)
    {
        Viewpoints = viewpoints;
        Warps = warps;
        Frustums = frustums;
    }

    public Tensor<T> Viewpoints { get; }

    public Tensor<T> Warps { get; }

    public Tensor<T> Frustums { get; }
}

/// <summary>
/// The DDIM schedule of SyncDreamer's sampler (<c>SyncDDIMSampler._make_schedule</c> over latent diffusion's
/// scaled-linear betas): uniform timesteps <c>range(0, T, T / steps) + 1</c>, their cumulative alphas, the previous
/// alphas (<c>alphas_cumprod[0]</c> first), and <c>sigma = eta sqrt((1 - a_prev) / (1 - a) (1 - a / a_prev))</c>.
/// </summary>
internal sealed class SyncDreamerDdimSchedule
{
    private SyncDreamerDdimSchedule(int[] timesteps, double[] alphas, double[] alphasPrevious, double[] sigmas)
    {
        Timesteps = timesteps;
        Alphas = alphas;
        AlphasPrevious = alphasPrevious;
        Sigmas = sigmas;
    }

    /// <summary>The DDIM timesteps, ascending.</summary>
    public int[] Timesteps { get; }

    public double[] Alphas { get; }

    public double[] AlphasPrevious { get; }

    public double[] Sigmas { get; }

    public int Count => Timesteps.Length;

    public static SyncDreamerDdimSchedule Create(int trainTimesteps, double betaStart, double betaEnd, int steps, double eta)
    {
        if (steps <= 0 || steps > trainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(steps), $"The DDIM steps must be in [1, {trainTimesteps}].");
        // make_beta_schedule("linear"): linspace(sqrt(start), sqrt(end), T) ** 2.
        var cumulative = new double[trainTimesteps];
        double product = 1.0, a = Math.Sqrt(betaStart), b = Math.Sqrt(betaEnd);
        for (int i = 0; i < trainTimesteps; i++)
        {
            double root = trainTimesteps == 1 ? a : a + (b - a) * i / (trainTimesteps - 1);
            product *= 1.0 - root * root;
            cumulative[i] = product;
        }

        // make_ddim_timesteps("uniform"): range(0, T, T // steps) + 1.
        int stride = trainTimesteps / steps;
        var timesteps = new List<int>();
        for (int t = 0; t < trainTimesteps; t += stride) timesteps.Add(t + 1);
        if (timesteps[timesteps.Count - 1] >= trainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(steps), $"{steps} DDIM steps over {trainTimesteps} timesteps step past the schedule.");
        int count = timesteps.Count;
        var alphas = new double[count];
        var previous = new double[count];
        var sigmas = new double[count];
        for (int i = 0; i < count; i++)
        {
            alphas[i] = cumulative[timesteps[i]];
            previous[i] = i == 0 ? cumulative[0] : cumulative[timesteps[i - 1]];
            sigmas[i] = eta * Math.Sqrt((1 - previous[i]) / (1 - alphas[i]) * (1 - alphas[i] / previous[i]));
        }

        return new SyncDreamerDdimSchedule(timesteps.ToArray(), alphas, previous, sigmas);
    }

    /// <summary>
    /// <c>denoise_apply_impl</c>: <c>x0 = (x - sqrt(1 - a) e) / sqrt(a)</c>,
    /// <c>x_prev = sqrt(a_prev) x0 + sqrt(max(1 - a_prev - sigma^2, 1e-7)) e + sigma z</c>, with no noise term at step 0.
    /// </summary>
    public Tensor<T> Step<T>(IEngine engine, Tensor<T> x, Tensor<T> noise, int index, Tensor<T>? fresh)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double a = Alphas[index], previous = AlphasPrevious[index], sigma = Sigmas[index];
        var predictedX0 = engine.TensorMultiplyScalar(
            engine.TensorSubtract(x, engine.TensorMultiplyScalar(noise, ops.FromDouble(Math.Sqrt(1 - a)))), ops.FromDouble(1.0 / Math.Sqrt(a)));
        var direction = engine.TensorMultiplyScalar(noise, ops.FromDouble(Math.Sqrt(Math.Max(1 - previous - sigma * sigma, 1e-7))));
        var next = engine.TensorAdd(engine.TensorMultiplyScalar(predictedX0, ops.FromDouble(Math.Sqrt(previous))), direction);
        return fresh is null ? next : engine.TensorAdd(next, engine.TensorMultiplyScalar(fresh, ops.FromDouble(sigma)));
    }
}
