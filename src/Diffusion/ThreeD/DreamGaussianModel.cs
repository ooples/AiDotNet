using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Segmentation.Salient;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// DreamGaussian: generative 3D Gaussian splatting with Score Distillation Sampling, mesh extraction and UV texture
/// refinement (Tang et al., ICLR 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stage 1 (<c>main.py</c>) starts from <see cref="DreamGaussianOptions.InitialGaussianCount"/> random Gaussians in a
/// ball of radius 0.5 and optimises them with Adam for <see cref="DreamGaussianOptions.Iterations"/> steps (500): each
/// step renders random orbit views and back-propagates the SDS gradient <c>w(t) (eps_hat - eps)</c> of a frozen 2D
/// prior through the VAE encoder and a differentiable tile rasterizer, with the timestep annealed from 980 to 20.
/// Image-to-3D uses Zero-1-to-3 (camera-conditioned, guidance 5, 256 x 256) plus RGB and mask losses on the input
/// view; text-to-3D uses Stable Diffusion (guidance 100, 512 x 512). Gaussians are densified, pruned and their
/// opacities reset on the 3DGS schedule.
/// </para>
/// <para>
/// The mesh is extracted from the Gaussians' density grid (<see cref="GaussianMeshExtractor"/>, marching cubes at
/// level 1). Stage 2 (<c>main2.py</c>) unwraps it into a UV atlas, back-projects 26 Gaussian renders into a 1024 x
/// 1024 texture, inpaints the seams, then refines the texture for <see cref="DreamGaussianOptions.RefineIterations"/>
/// steps (50) against SDEdit-denoised renders of the mesh (strength 0.8 to 0.95), plus the reference view in
/// image-to-3D.
/// </para>
/// <para>
/// <b>For Beginners:</b> DreamGaussian turns one picture (or a sentence) into a textured 3D model. It first grows a
/// cloud of soft coloured blobs until pictures of it look right to a 2D diffusion model from every side, then turns
/// the cloud into a mesh and paints the mesh's texture.
/// </para>
/// <para>
/// The 2D priors are networks of this library and start untrained; load pretrained weights into
/// <see cref="ImageGuidanceModel"/> (Zero-1-to-3) and this model's U-Net, VAE and conditioner (Stable Diffusion) for
/// meaningful shapes, or supply any prior through <see cref="ImageGuidanceFactory"/> / <see cref="TextGuidanceFactory"/>.
/// </para>
/// <para>
/// Reference: Tang et al., "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation", ICLR 2024
/// (arXiv 2309.16653); official code github.com/dreamgaussian/dreamgaussian.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dreamGaussian = new DreamGaussianModel&lt;float&gt;(seed: 0);
/// var result = dreamGaussian.RunImageTo3D(rgbaImage);   // [1, 4, H, W] in [0, 1]
/// var mesh = result.FinalMesh;                          // UV-textured mesh
/// var points = result.PointCloud;                       // [1, N, 6] Gaussian centres and colours
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation", "https://arxiv.org/abs/2309.16653", Year = 2024, Authors = "Tang et al.")]
public partial class DreamGaussianModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
    }

    #region Constants

    /// <summary>
    /// Number of latent channels (4, standard SD VAE).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension from CLIP text encoder (768).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Base channel count for the SD 1.5 U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// Initial number of Gaussians (<c>num_pts</c> = 5000 in the official configs).
    /// </summary>
    private const int DEFAULT_POINT_COUNT = 5000;

    #endregion

    #region Fields

    /// <summary>
    /// The SD 1.5 U-Net noise predictor for SDS guidance.
    /// </summary>
    private UNetNoisePredictor<T>? _unet;

    /// <summary>
    /// The standard SD VAE for encoding/decoding rendered views.
    /// </summary>
    private StandardVAE<T>? _vae;

    /// <summary>
    /// The CLIP text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;
    private readonly int? _seed;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor { get { EnsureInitialized(); return _unet; } }

    /// <inheritdoc />
    public override IVAEModel<T> VAE { get { EnsureInitialized(); return _vae; } }

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <inheritdoc />
    public override bool SupportsTexture => true;

    /// <inheritdoc />
    public override bool SupportsNovelView => true;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => true;

    /// <summary>
    /// The pipeline settings; every default is the official code's. The generation methods' arguments override the
    /// corresponding settings for one call.
    /// </summary>
    public DreamGaussianOptions Settings { get; set; }

    /// <summary>
    /// The Zero-1-to-3 model of the image-to-3D guidance. Null creates a default one on first use. It is used, not
    /// owned: its weights are not part of this model's parameters.
    /// </summary>
    [ExternalState]
    public Zero123Model<T>? ImageGuidanceModel { get; set; }

    /// <summary>
    /// Builds the image-to-3D guidance from the reference image ([1, 3, S, S] on white) and the run's random source.
    /// Null uses Zero-1-to-3 (<see cref="ImageGuidanceModel"/>).
    /// </summary>
    public Func<Tensor<T>, Random, IScoreDistillationGuidance<T>>? ImageGuidanceFactory { get; set; }

    /// <summary>
    /// Builds the text-to-3D guidance from the prompt and negative prompt. Null uses Stable Diffusion with this
    /// model's U-Net, VAE, scheduler and conditioner.
    /// </summary>
    public Func<string, string?, IScoreDistillationGuidance<T>>? TextGuidanceFactory { get; set; }

    /// <summary>
    /// Separates the object from the background of an RGB input given without a mask. Null creates the reference's
    /// choice on first use: U²-Net run as rembg runs it (<see cref="U2NetForegroundSegmenter{T}"/>). Load the released
    /// <c>u2net.pth</c> into its <see cref="U2NetForegroundSegmenter{T}.Network"/> for rembg's masks.
    /// </summary>
    [ExternalState]
    public IForegroundSegmenter<T>? ForegroundSegmenter { get; set; }

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of DreamGaussianModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses SD 1.5 defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">Noise scheduler. If null, uses DDIM.</param>
    /// <param name="unet">Custom U-Net. If null, creates SD 1.5 backbone for SDS.</param>
    /// <param name="vae">Custom VAE. If null, creates standard SD VAE.</param>
    /// <param name="conditioner">CLIP text encoder conditioning module (needed for text-to-3D).</param>
    /// <param name="defaultPointCount">Initial number of Gaussians (official: 5000).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DreamGaussianModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultPointCount = DEFAULT_POINT_COUNT,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultPointCount,
            architecture)
    {
        _conditioner = conditioner;
        _seed = seed;
        Settings = new DreamGaussianOptions { InitialGaussianCount = defaultPointCount };

        if (unet is not null || vae is not null)
            InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void EnsureInitialized()
    {
        if (_unet is null || _vae is null)
            InitializeLayers(null, null, _seed);
    }

    /// <summary>
    /// Initializes the U-Net and VAE layers using custom or default configurations.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
    }

    #endregion

    #region DreamGaussian Pipeline

    /// <summary>
    /// Image-to-3D with <see cref="Settings"/>: stage 1, mesh extraction and (when enabled) the texture stage.
    /// </summary>
    /// <param name="image">The input image in [0, 1]: [1, 4, H, W] RGBA or [1, 3, H, W] RGB ([C, H, W] is accepted
    /// too).</param>
    /// <param name="mask">
    /// Optional foreground mask with H x W values in [0, 1]. The mask is taken from, in order: this argument, the
    /// image's alpha channel, then <see cref="ForegroundSegmenter"/> (U²-Net, as the reference's <c>rembg</c>
    /// pre-processing).
    /// </param>
    /// <param name="seed">Random seed; null uses the model's seed, or a secure random source when that is null too.</param>
    /// <returns>The Gaussians, point cloud, mesh and textured mesh.</returns>
    public DreamGaussianResult<T> RunImageTo3D(Tensor<T> image, Tensor<T>? mask = null, int? seed = null)
        => RunImage(image, mask, Settings, seed, includeMesh: true);

    /// <summary>Text-to-3D with <see cref="Settings"/>.</summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt; null uses the unconditional embedding.</param>
    /// <param name="seed">Random seed; null uses the model's seed, or a secure random source when that is null too.</param>
    /// <returns>The Gaussians, point cloud, mesh and textured mesh.</returns>
    public DreamGaussianResult<T> RunTextTo3D(string prompt, string? negativePrompt = null, int? seed = null)
        => RunText(prompt, negativePrompt, Settings, seed, includeMesh: true);

    private DreamGaussianResult<T> RunImage(Tensor<T> image, Tensor<T>? mask, DreamGaussianOptions settings, int? seed, bool includeMesh)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (settings is null) throw new InvalidOperationException("Settings must not be null.");
        var random = CreateRandom(seed);
        var (reference, referenceMask) = PrepareReference(
            image, mask, settings, () => ForegroundSegmenter ??= new U2NetForegroundSegmenter<T>(seed: _seed));
        var referenceTensor = DreamGaussianPipeline<T>.ToBatch(new[] { reference }, settings.ReferenceSize);
        var guidance = ImageGuidanceFactory is not null
            ? ImageGuidanceFactory(referenceTensor, random)
            : new Zero123ScoreDistillationGuidance<T>(
                ImageGuidanceModel ??= new Zero123Model<T>(seed: _seed), referenceTensor, random, settings.ImageGuidanceScale,
                settings.ImageGuidanceResolution, settings.MinTimestepRatio, settings.MaxTimestepRatio);
        var pipeline = new DreamGaussianPipeline<T>(settings, DreamGaussianMode.ImageTo3D, guidance, reference, referenceMask, random);
        return includeMesh ? pipeline.Run() : pipeline.RunGaussiansOnly();
    }

    private DreamGaussianResult<T> RunText(string prompt, string? negativePrompt, DreamGaussianOptions settings, int? seed, bool includeMesh)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        if (settings is null) throw new InvalidOperationException("Settings must not be null.");
        var random = CreateRandom(seed);
        IScoreDistillationGuidance<T> guidance;
        if (TextGuidanceFactory is not null)
        {
            guidance = TextGuidanceFactory(prompt, negativePrompt);
        }
        else
        {
            var conditioner = Conditioner
                ?? throw new InvalidOperationException(
                    "Text-to-3D needs a text conditioner: pass one to the constructor or set TextGuidanceFactory.");
            var promptEmbedding = conditioner.EncodeText(conditioner.Tokenize(prompt));
            var negativeEmbedding = string.IsNullOrEmpty(negativePrompt)
                ? conditioner.GetUnconditionalEmbedding(1)
                : conditioner.EncodeText(conditioner.Tokenize(negativePrompt ?? string.Empty));
            guidance = new StableDiffusionScoreDistillationGuidance<T>(
                NoisePredictor, VAE, Scheduler, promptEmbedding, negativeEmbedding, settings.TextGuidanceScale,
                settings.TextGuidanceResolution, settings.MinTimestepRatio, settings.MaxTimestepRatio);
        }

        var pipeline = new DreamGaussianPipeline<T>(settings, DreamGaussianMode.TextTo3D, guidance, null, null, random);
        return includeMesh ? pipeline.Run() : pipeline.RunGaussiansOnly();
    }

    private Random CreateRandom(int? seed)
    {
        int? effective = seed ?? _seed;
        return effective.HasValue ? RandomHelper.CreateSeededRandom(effective.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// The reference's input handling: the foreground alpha (the caller's mask, else the RGBA alpha, else the
    /// segmenter, as <c>process.py</c>'s <c>rembg.remove</c>), <c>process.py</c>'s recentring (when
    /// <see cref="DreamGaussianOptions.RecenterReference"/>, straight to the reference size: <c>process.py</c>'s
    /// <c>--size</c> and <c>ref_size</c> are both 256), then <c>load_input</c>: the RGBA image resized to
    /// <see cref="DreamGaussianOptions.ReferenceSize"/> (area filter when shrinking) and the colour composited onto
    /// white.
    /// </summary>
    /// <returns>The image [S, S, 3] on white and the mask [S, S].</returns>
    internal static (double[] Image, double[] Mask) PrepareReference(
        Tensor<T> image, Tensor<T>? mask, DreamGaussianOptions settings, Func<IForegroundSegmenter<T>> segmenter)
    {
        var batch = image.Rank == 3 ? image.Reshape(new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] }) : image;
        if (batch.Rank != 4 || batch.Shape[0] != 1 || (batch.Shape[1] != 3 && batch.Shape[1] != 4))
            throw new ArgumentException("The input image must be [1, 3, H, W] or [1, 4, H, W] (or without the batch axis).", nameof(image));
        int height = batch.Shape[2], width = batch.Shape[3], sourcePixels = height * width;
        var source = batch.Cast<double>().ToArray();
        double[] alpha;
        if (mask is not null)
        {
            if (mask.Length != sourcePixels)
                throw new ArgumentException($"The mask must have {height} x {width} values.", nameof(mask));
            alpha = mask.Cast<double>().ToArray();
        }
        else if (batch.Shape[1] == 4)
        {
            alpha = new double[sourcePixels];
            Array.Copy(source, 3 * sourcePixels, alpha, 0, sourcePixels);
        }
        else
        {
            var matte = segmenter().Segment(batch);
            if (matte is null || matte.Length != sourcePixels)
                throw new InvalidOperationException("The foreground segmenter must return an H x W mask.");
            alpha = matte.Cast<double>().ToArray();
        }

        var rgba = new double[4 * sourcePixels];
        Array.Copy(source, 0, rgba, 0, 3 * sourcePixels);
        for (int p = 0; p < sourcePixels; p++)
            rgba[(3 * sourcePixels) + p] = Math.Min(1.0, Math.Max(0.0, alpha[p]));

        int size = settings.ReferenceSize;
        var data = rgba;
        if (settings.RecenterReference)
        {
            var recentered = DreamGaussianPreprocessing.Recenter(rgba, height, width, size, settings.RecenterBorderRatio);
            if (recentered is not null)
            {
                data = recentered;
                height = size;
                width = size;
            }
        }

        if (height != size || width != size)
        {
            bool shrinking = height >= size && width >= size;
            var tensor = new Tensor<double>(new[] { 1, 4, height, width }, new Vector<double>(data)).Cast<T>();
            data = AiDotNetEngine.Current.Interpolate(
                tensor, new[] { size, size }, shrinking ? InterpolateMode.Area : InterpolateMode.Bilinear, false)
                .Cast<double>().ToArray();
        }

        int pixels = size * size;
        var rgb = new double[3 * pixels];
        var referenceMask = new double[pixels];
        for (int p = 0; p < pixels; p++)
        {
            double a = Math.Min(1.0, Math.Max(0.0, data[(3 * pixels) + p]));
            referenceMask[p] = a;
            for (int c = 0; c < 3; c++)
                rgb[(3 * p) + c] = (data[(c * pixels) + p] * a) + (1.0 - a);
        }

        return (rgb, referenceMask);
    }

    private static DreamGaussianOptions CopySettings(DreamGaussianOptions settings)
    {
        if (settings is null) throw new InvalidOperationException("Settings must not be null.");
        var copy = settings.ShallowCopy();
        var extraction = settings.MeshExtraction ?? new GaussianMeshExtractionOptions();
        copy.MeshExtraction = new GaussianMeshExtractionOptions
        {
            Resolution = extraction.Resolution,
            BlockCount = extraction.BlockCount,
            RelaxRatio = extraction.RelaxRatio,
            DensityThreshold = extraction.DensityThreshold,
            OpacityCullThreshold = extraction.OpacityCullThreshold,
            NormalizedHalfExtent = extraction.NormalizedHalfExtent,
        };
        return copy;
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Image-to-3D: the textured mesh of <see cref="RunImageTo3D"/>.
    /// </summary>
    /// <param name="inputImage">The input image, as for <see cref="RunImageTo3D"/>.</param>
    /// <param name="numViews">Novel views per stage-1 and stage-2 step (<c>batch_size</c>; official: 1).</param>
    /// <param name="numInferenceSteps">Denoising steps of each stage-2 SDEdit refinement (official: 50).</param>
    /// <param name="guidanceScale">Zero-1-to-3 classifier-free guidance scale (official: 5).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>The textured mesh (the extracted mesh when the texture stage is off).</returns>
    public override Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = 1,
        int numInferenceSteps = 50,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        var settings = CopySettings(Settings);
        settings.BatchSize = numViews;
        settings.RefineDenoisingSteps = numInferenceSteps;
        settings.ImageGuidanceScale = guidanceScale;
        return RunImage(inputImage, null, settings, seed, includeMesh: true).FinalMesh;
    }

    /// <summary>
    /// Text-to-3D point cloud: the centres and colours of the stage-1 Gaussians.
    /// </summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt.</param>
    /// <param name="numPoints">Initial number of Gaussians (null: <see cref="DreamGaussianOptions.InitialGaussianCount"/>);
    /// densification and pruning change the final count.</param>
    /// <param name="numInferenceSteps">Stage-1 optimisation steps (official: 500).</param>
    /// <param name="guidanceScale">Stable Diffusion classifier-free guidance scale (official: 100).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>[1, N, 6] (x, y, z, r, g, b).</returns>
    public override Tensor<T> GeneratePointCloud(
        string prompt,
        string? negativePrompt = null,
        int? numPoints = null,
        int numInferenceSteps = 500,
        double guidanceScale = 100.0,
        int? seed = null)
    {
        var settings = CopySettings(Settings);
        if (numPoints.HasValue) settings.InitialGaussianCount = numPoints.Value;
        settings.Iterations = numInferenceSteps;
        settings.TextGuidanceScale = guidanceScale;
        return RunText(prompt, negativePrompt, settings, seed, includeMesh: false).PointCloud;
    }

    /// <summary>
    /// Text-to-3D mesh: the textured mesh of <see cref="RunTextTo3D"/>.
    /// </summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt.</param>
    /// <param name="resolution">Density grid resolution of the mesh extraction (official: 128).</param>
    /// <param name="numInferenceSteps">Stage-1 optimisation steps (official: 500).</param>
    /// <param name="guidanceScale">Stable Diffusion classifier-free guidance scale (official: 100).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>The textured mesh (the extracted mesh when the texture stage is off).</returns>
    public override Mesh3D<T> GenerateMesh(
        string prompt,
        string? negativePrompt = null,
        int resolution = 128,
        int numInferenceSteps = 500,
        double guidanceScale = 100.0,
        int? seed = null)
    {
        var settings = CopySettings(Settings);
        settings.MeshExtraction.Resolution = resolution;
        settings.Iterations = numInferenceSteps;
        settings.TextGuidanceScale = guidanceScale;
        return RunText(prompt, negativePrompt, settings, seed, includeMesh: true).FinalMesh;
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var settings = Settings ?? new DreamGaussianOptions();
        var metadata = new ModelMetadata<T>
        {
            Name = "DreamGaussian",
            Version = "1.0",
            Description = "DreamGaussian generative Gaussian splatting with SDS, mesh extraction and UV texture refinement",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "gaussian-splatting-sds");
        metadata.SetProperty("representation", "3d-gaussian-splatting");
        metadata.SetProperty("guidance_backbone", "SD-1.5 (text), Zero-1-to-3 (image)");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("optimization_steps", settings.Iterations);
        metadata.SetProperty("refinement_steps", settings.RefineIterations);
        metadata.SetProperty("mesh_extraction", "marching-cubes");
        metadata.SetProperty("texture_refinement", "uv-space-sdedit-mse");
        metadata.SetProperty("texture_size", settings.TextureSize);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_points", settings.InitialGaussianCount);

        return metadata;
    }

    #endregion
}
