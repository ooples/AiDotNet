using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// MVDream - Multi-View Diffusion Model for 3D-consistent image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MVDream is a multi-view diffusion model that generates 3D-consistent images
/// from multiple viewpoints simultaneously. It enables high-quality 3D generation
/// by leveraging multi-view supervision during training.
///
/// Key capabilities:
/// 1. Multi-View Generation: Generate multiple consistent views of an object
/// 2. Text-to-3D: Create 3D content from text descriptions
/// 3. Image-to-3D: Convert single images to 3D representations
/// 4. Score Distillation Sampling (SDS): Guide NeRF/3DGS optimization
/// 5. Novel View Synthesis: Generate unseen viewpoints of objects
/// </para>
/// <para>
/// <b>For Beginners:</b> MVDream creates multiple images of the same object
/// from different angles, all consistent with each other:
///
/// Example: "A red sports car"
/// - Front view: shows the car's front grille and headlights
/// - Side view: shows the profile with wheels and doors
/// - Back view: shows tail lights and rear design
/// - All views are consistent (same car, same color, same features)
///
/// This consistency is what enables 3D reconstruction:
/// - Multiple views + triangulation = 3D model
/// - Can be used with Score Distillation for high-quality 3D
/// </para>
/// <para>
/// Technical specifications:
/// - Image resolution: 256x256 per view (default)
/// - Number of views: 4 (orthogonal) or 8 (comprehensive)
/// - Latent channels: 4 (Stable Diffusion compatible)
/// - Context dimension: 1024 (CLIP/T5 embeddings)
/// - Camera model: Spherical coordinates (azimuth, elevation, radius)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var nerFRenderOutput = Tensor&lt;float&gt;.CreateRandom(1, 3, 32, 32);
/// // Create MVDream model
/// var mvdream = new MVDreamModel&lt;float&gt;();
///
/// // Generate 4 views of an object
/// var views = mvdream.GenerateMultiView(
///     prompt: "A cute robot toy",
///     numViews: 4,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate for score distillation (SDS)
/// var sdsGradient = mvdream.ComputeScoreDistillationGradients(
///     renderedViews: nerFRenderOutput,
///     prompt: "A cute robot toy",
///     timestep: 500,
///     guidanceScale: 100.0);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MVDream: Multi-view Diffusion for 3D Generation", "https://arxiv.org/abs/2308.16512", Year = 2024, Authors = "Shi et al.")]
public partial class MVDreamModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Each module is registered once. The base U-Net already reaches the surface through the
    /// <see cref="NoisePredictor"/> slot (and the VAE through <see cref="VAE"/>), so the multi-view U-Net contributes
    /// only the modules outside it here; registering the whole multi-view U-Net as well counted every base U-Net
    /// weight twice. Order: base U-Net, VAE, camera embedding, then ImageDream's resampler and the CLIP image encoder
    /// when present.
    /// </remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_imageVAE);
        RegisterParameterComponent(_multiViewUNet.CameraEmbedding);
        RegisterParameterComponent(_multiViewUNet.ImageEmbed);
        RegisterParameterComponent(_imageEncoder);
    }

    #region Constants

    /// <summary>
    /// Default image resolution for each view.
    /// </summary>
    /// <remarks>256x256 per view balances quality and memory for multi-view generation.</remarks>
    public const int MVDREAM_IMAGE_SIZE = 256;

    /// <summary>
    /// MVDream latent channels (Stable Diffusion compatible).
    /// </summary>
    /// <remarks>4 latent channels match the standard SD VAE architecture for compatibility.</remarks>
    public const int MVDREAM_LATENT_CHANNELS = 4;

    /// <summary>
    /// MVDream base channels for U-Net.
    /// </summary>
    /// <remarks>320 base channels match the SD 1.5 U-Net architecture that MVDream extends.</remarks>
    public const int MVDREAM_BASE_CHANNELS = 320;

    /// <summary>
    /// Context dimension for text/image conditioning.
    /// </summary>
    /// <remarks>1024 accommodates CLIP/T5 text embeddings plus camera position information.</remarks>
    public const int MVDREAM_CONTEXT_DIM = 1024;

    /// <summary>
    /// Default camera distance from object.
    /// </summary>
    /// <remarks>1.5 units provides a good framing for unit-scale objects centered at the origin.</remarks>
    public const double DEFAULT_CAMERA_DISTANCE = 1.5;

    #endregion

    #region Fields

    /// <summary>
    /// The multi-view aware U-Net noise predictor.
    /// </summary>
    private MultiViewUNet<T> _multiViewUNet;

    /// <summary>
    /// The VAE for image encoding/decoding.
    /// </summary>
    private StandardVAE<T> _imageVAE;

    /// <summary>
    /// Text conditioning module (CLIP/T5).
    /// </summary>
    private readonly IConditioningModule<T>? _textConditioner;

    /// <summary>
    /// Image conditioning module for image-to-3D.
    /// </summary>
    private readonly IConditioningModule<T>? _imageConditioner;

    /// <summary>
    /// ImageDream's CLIP image encoder (open_clip ViT-H/14), whose penultimate hidden states the resampler reads.
    /// </summary>
    private readonly ClipVisionEncoder<T>? _imageEncoder;

    /// <summary>
    /// Model configuration.
    /// </summary>
    private readonly MVDreamConfig _config;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _multiViewUNet.BaseUNet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _imageVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _textConditioner;

    /// <inheritdoc />
    public override int LatentChannels => MVDREAM_LATENT_CHANNELS;

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
    /// Gets the image conditioner for image-to-3D tasks.
    /// </summary>
    public IConditioningModule<T>? ImageConditioner => _imageConditioner;

    /// <summary>
    /// Gets the camera embedding module (the multi-view U-Net's <c>camera_embed</c>).
    /// </summary>
    public CameraEmbedding<T> CameraEmbedding => _multiViewUNet.CameraEmbedding;

    /// <summary>
    /// Gets ImageDream's CLIP image encoder, or null when the image path runs without one.
    /// </summary>
    public ClipVisionEncoder<T>? ImageEncoder => _imageEncoder;

    /// <summary>
    /// Gets the multi-view U-Net.
    /// </summary>
    public MultiViewUNet<T> MultiViewUNet => _multiViewUNet;

    /// <summary>
    /// Gets the model configuration.
    /// </summary>
    public MVDreamConfig Config => _config;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new MVDream model with full customization support.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="multiViewUNet">Optional custom multi-view U-Net.</param>
    /// <param name="imageVAE">Optional custom image VAE.</param>
    /// <param name="textConditioner">Optional text conditioning module.</param>
    /// <param name="imageConditioner">Optional image conditioning module: an alternative source of the CLIP image
    /// features ImageDream's resampler reads, returning <c>[1, tokens, ImageFeatureDim]</c>.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="imageEncoder">ImageDream's CLIP image encoder (<see cref="ClipVisionPreset.ViTH14OpenClip"/>), whose
    /// penultimate hidden states feed the resampler; used when the multi-view U-Net has ImageDream's image prompt.</param>
    public MVDreamModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MultiViewUNet<T>? multiViewUNet = null,
        StandardVAE<T>? imageVAE = null,
        IConditioningModule<T>? textConditioner = null,
        IConditioningModule<T>? imageConditioner = null,
        MVDreamConfig? config = null,
        int? seed = null,
        ClipVisionEncoder<T>? imageEncoder = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultPointCount: 8192,
            architecture: architecture)
    {
        _config = config ?? MVDreamConfig.Default;
        _textConditioner = textConditioner;
        _imageConditioner = imageConditioner;
        _imageEncoder = imageEncoder;

        InitializeLayers(multiViewUNet, imageVAE, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the model layers, using provided components or creating defaults.
    /// </summary>
    [MemberNotNull(nameof(_multiViewUNet), nameof(_imageVAE))]
    private void InitializeLayers(
        MultiViewUNet<T>? multiViewUNet,
        StandardVAE<T>? imageVAE,
        int? seed)
    {
        _imageVAE = imageVAE ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: MVDREAM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        _multiViewUNet = multiViewUNet ?? new MultiViewUNet<T>(
            inputChannels: MVDREAM_LATENT_CHANNELS,
            outputChannels: MVDREAM_LATENT_CHANNELS,
            baseChannels: MVDREAM_BASE_CHANNELS,
            numViews: _config.DefaultNumViews,
            contextDim: MVDREAM_CONTEXT_DIM,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates multiple consistent views from a text prompt, sampled the way the official <c>scripts/t2i.py</c> does.
    /// </summary>
    /// <param name="prompt">Text description of the object.</param>
    /// <param name="negativePrompt">Optional negative prompt; the unconditional embedding when null or empty.</param>
    /// <param name="numViews">Number of views to generate (4 in MVDream).</param>
    /// <param name="numInferenceSteps">Number of denoising steps (50 in <c>t2i.py</c>).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (10 in <c>t2i.py</c>).</param>
    /// <param name="elevation">Camera elevation in degrees, positive looking down; null for
    /// <see cref="MVDreamConfig.DefaultElevation"/> (15).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="azimuthStart">The first view's azimuth in degrees; null for
    /// <see cref="MVDreamConfig.DefaultAzimuthStart"/> (90). View v is at <c>azimuthStart + v · AzimuthSpan / numViews</c>.</param>
    /// <returns>The generated views, each [1, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// The cameras are <c>get_camera(numViews, elevation, azimuth_start, azimuth_span)</c>: unit-sphere camera-to-world
    /// matrices in Blender's axes, flattened, embedded by <c>camera_embed</c> and added to the time embedding. All the
    /// views run through the U-Net as one batch, so its 3D self-attention attends across them; the conditional and
    /// unconditional branches of guidance are separate calls.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This generates multiple pictures of an object from different angles, all looking
    /// consistent. With four views they are 90 degrees apart; a higher elevation looks more from above.
    /// </para>
    /// </remarks>
    public virtual Tensor<T>[] GenerateMultiView(
        string prompt,
        string? negativePrompt = null,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 10.0,
        double? elevation = null,
        int? seed = null,
        double? azimuthStart = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Multi-view generation requires a text conditioning module.");
        if (numViews <= 0) throw new ArgumentOutOfRangeException(nameof(numViews), "At least one view is required.");
        RequireTextOnlyUNet("Text-to-multi-view generation");

        var cameras = CameraEmbedding<T>.CameraTensor(CameraEmbedding<T>.GetCameras(
            numViews, elevation ?? _config.DefaultElevation, azimuthStart ?? _config.DefaultAzimuthStart,
            _config.AzimuthSpan));

        var promptEmbedding = _textConditioner.EncodeText(_textConditioner.Tokenize(prompt));
        bool guided = guidanceScale > 1.0;
        Tensor<T>? negativeEmbedding = null;
        if (guided)
        {
            negativeEmbedding = string.IsNullOrEmpty(negativePrompt)
                ? _textConditioner.GetUnconditionalEmbedding(1)
                : _textConditioner.EncodeText(_textConditioner.Tokenize(negativePrompt ?? string.Empty));
        }

        var conditions = RepeatCondition(promptEmbedding, numViews);
        var negativeConditions = RepeatCondition(negativeEmbedding, numViews);

        var latentShape = new[] { numViews, MVDREAM_LATENT_CHANNELS, MVDREAM_IMAGE_SIZE / 8, MVDREAM_IMAGE_SIZE / 8 };
        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);
        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = _multiViewUNet.PredictNoiseMultiView(latents, timestep, conditions, cameras);
            if (negativeEmbedding is not null)
            {
                var unconditional = _multiViewUNet.PredictNoiseMultiView(latents, timestep, negativeConditions, cameras);
                noisePrediction = ApplyGuidance(unconditional, noisePrediction, guidanceScale);
            }

            var latentVector = Scheduler.Step(noisePrediction.ToVector(), timestep, latents.ToVector(), NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode each view (unscale latent before VAE decoding)
        var views = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var viewLatent = ExtractView(latents, v);
            views[v] = _imageVAE.Decode(_imageVAE.UnscaleLatent(viewLatent));
        }

        return views;
    }

    /// <summary>
    /// Rejects a text-only call on ImageDream's U-Net: its decoupled cross-attention always reads the last
    /// <see cref="MultiViewUNetOptions.ImagePromptTokens"/> context tokens as the image prompt
    /// (<c>context[:, -ip_dim:]</c> in the official attention), so it is driven from an image.
    /// </summary>
    private void RequireTextOnlyUNet(string operation)
    {
        if (_multiViewUNet.HasImagePrompt)
            throw new InvalidOperationException(
                $"{operation} needs MVDream's text-only U-Net; this U-Net is ImageDream's, whose cross-attention reads " +
                "image-prompt tokens after the text. Use GenerateNovelViewsFromImage.");
    }

    /// <summary>
    /// Generates 3D from a single input image.
    /// </summary>
    /// <param name="inputImage">Input image tensor [1, channels, height, width].</param>
    /// <param name="prompt">Optional text guidance.</param>
    /// <param name="numViews">Number of novel views to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated 3D mesh.</returns>
    public override Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        // Generate novel views from input image
        var views = GenerateNovelViewsFromImage(
            inputImage,
            numViews,
            numInferenceSteps,
            guidanceScale,
            seed);

        // Reconstruct 3D from multi-view images, seen from the cameras that generated them (the image path's views
        // start at azimuth 0, the input's viewpoint).
        var cameraPositions = GenerateCameraPositions(numViews, _config.ImageElevation);
        return ReconstructFromMultiView(views, cameraPositions);
    }

    /// <summary>
    /// Generates novel views from a single input image the way ImageDream (Wang and Shi, 2023, arXiv 2312.02201;
    /// official <c>scripts/demo.py</c>) does.
    /// </summary>
    /// <param name="inputImage">Input image [1, 3, H, W] in the VAE's range: the object seen from azimuth 0.</param>
    /// <param name="numViews">Number of views to generate, at azimuths <c>v · AzimuthSpan / numViews</c> degrees.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (5.0 in ImageDream's demo). Above 1 the
    /// unconditional branch has the image inputs zeroed, as ImageDream's demo does.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="elevation">Camera elevation in degrees (positive looks down); null for
    /// <see cref="MVDreamConfig.ImageElevation"/> (5).</param>
    /// <returns>The <paramref name="numViews"/> generated views, each [1, 3, H, W].</returns>
    /// <remarks>
    /// <para>
    /// The pixel controller: the clean VAE latent of the input image is appended as an extra frame after the generated
    /// views, with ImageDream's all-zero camera (<c>get_camera(..., extra_view=True)</c>), and written back into that
    /// frame before every denoiser call (<c>ip_img</c> at <c>(num_frames - 1)::num_frames</c>), so the 3D
    /// self-attention lets every generated view see the image. The extra frame is dropped from the result.
    /// </para>
    /// <para>
    /// The local controller, when the U-Net is ImageDream's (<see cref="MultiViewUNet{T}.HasImagePrompt"/>): the
    /// penultimate hidden states of the CLIP image encoder (<see cref="ImageEncoder"/>, open_clip ViT-H/14; or the
    /// <see cref="ImageConditioner"/>'s features) go through the resampler (<c>image_embed</c>) into
    /// <see cref="MultiViewUNetOptions.ImagePromptTokens"/> tokens that follow the empty prompt's text tokens, and every
    /// transformer's decoupled cross-attention adds their attention, scaled by
    /// <see cref="MultiViewUNetOptions.ImagePromptWeight"/>, to the text attention. The unconditional branch is the
    /// unconditional text, the resampled all-zero features and an all-zero image frame. With MVDream's text-only U-Net
    /// the image enters through the pixel controller alone.
    /// </para>
    /// </remarks>
    public virtual Tensor<T>[] GenerateNovelViewsFromImage(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 5.0,
        int? seed = null,
        double? elevation = null)
    {
        if (inputImage is null) throw new ArgumentNullException(nameof(inputImage));
        if (numViews <= 0) throw new ArgumentOutOfRangeException(nameof(numViews), "At least one view is required.");

        var engine = AiDotNetEngine.Current;
        var imageLatent = _imageVAE.ScaleLatent(_imageVAE.Encode(inputImage, sampleMode: false));
        var emptyFrame = new Tensor<T>(imageLatent.Shape.ToArray());
        int frames = numViews + 1;

        // The generated views' cameras from the input's azimuth, then ImageDream's extra image view (all-zero camera).
        var cameras = CameraEmbedding<T>.CameraTensor(CameraEmbedding<T>.GetCameras(
            numViews, elevation ?? _config.ImageElevation, 0.0, _config.AzimuthSpan, extraView: true));

        bool guided = guidanceScale > 1.0;
        var (context, unconditionalContext) = ImagePromptContexts(inputImage, guided);
        var conditions = RepeatCondition(context, frames);
        var unconditionalConditions = RepeatCondition(unconditionalContext, frames);

        var latentShape = new[] { frames, imageLatent.Shape[1], imageLatent.Shape[2], imageLatent.Shape[3] };
        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);
        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = _multiViewUNet.PredictNoiseMultiView(
                WithLastFrame(engine, latents, numViews, imageLatent), timestep, conditions, cameras);
            if (guided)
            {
                var unconditional = _multiViewUNet.PredictNoiseMultiView(
                    WithLastFrame(engine, latents, numViews, emptyFrame), timestep, unconditionalConditions, cameras);
                noisePrediction = ApplyGuidance(unconditional, noisePrediction, guidanceScale);
            }

            var latentVector = Scheduler.Step(noisePrediction.ToVector(), timestep, latents.ToVector(), NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        var views = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var viewLatent = ExtractView(latents, v);
            views[v] = _imageVAE.Decode(_imageVAE.UnscaleLatent(viewLatent));
        }

        return views;
    }

    /// <summary>
    /// The cross-attention contexts of the image path. With ImageDream's U-Net: the empty prompt's text tokens followed
    /// by the resampled image tokens, and, for guidance, the unconditional text tokens followed by the resampled
    /// all-zero features (the demo's <c>uc["ip"] = zeros_like(ip_embed)</c>). With MVDream's text-only U-Net: the empty
    /// prompt's text tokens (null without a text conditioner).
    /// </summary>
    private (Tensor<T>? Conditional, Tensor<T>? Unconditional) ImagePromptContexts(Tensor<T> inputImage, bool guided)
    {
        Tensor<T>? text = null;
        Tensor<T>? negative = null;
        if (_textConditioner is not null)
        {
            text = _textConditioner.EncodeText(_textConditioner.Tokenize(string.Empty));
            if (guided) negative = _textConditioner.GetUnconditionalEmbedding(1);
        }

        if (!_multiViewUNet.HasImagePrompt)
        {
            if (_imageEncoder is not null || _imageConditioner is not null)
                throw new InvalidOperationException(
                    "The image encoder's features feed ImageDream's decoupled image cross-attention, which this " +
                    "multi-view U-Net does not have; build it with MultiViewUNetOptions.ImageDream().");
            return (text, negative);
        }

        if (text is null)
            throw new InvalidOperationException(
                "ImageDream's image path conditions on the encoded empty prompt followed by the image tokens; give the " +
                "model a text conditioning module.");

        var engine = AiDotNetEngine.Current;
        var features = ImageFeatures(inputImage);
        var conditional = engine.TensorConcatenate(new[] { AsTokens(text), _multiViewUNet.EmbedImagePrompt(features) }, 1);
        if (!guided || negative is null) return (conditional, null);

        var zeroTokens = _multiViewUNet.EmbedImagePrompt(new Tensor<T>(features.Shape.ToArray()));
        return (conditional, engine.TensorConcatenate(new[] { AsTokens(negative), zeroTokens }, 1));
    }

    /// <summary>
    /// The CLIP image features ImageDream's resampler reads, <c>[1, tokens, ImageFeatureDim]</c>: the image encoder's
    /// penultimate hidden states (<c>FrozenOpenCLIPImageEmbedder</c> with <c>layer="penultimate"</c>), or the image
    /// conditioner's output.
    /// </summary>
    private Tensor<T> ImageFeatures(Tensor<T> inputImage)
    {
        Tensor<T> features;
        if (_imageEncoder is not null)
            features = _imageEncoder.EncodePenultimateHiddenStates(inputImage);
        else if (_imageConditioner is not null)
            features = _imageConditioner.Encode(inputImage);
        else
            throw new InvalidOperationException(
                "ImageDream's resampler reads CLIP image features; give the model a ClipVisionEncoder " +
                "(ClipVisionPreset.ViTH14OpenClip) or an image conditioning module.");

        int width = _multiViewUNet.Options.ImageFeatureDim;
        if (features.Rank != 3 || features.Shape[0] != 1 || features.Shape[2] != width)
            throw new InvalidOperationException(
                $"The image features must be [1, tokens, {width}]; got [{string.Join(", ", features.Shape.ToArray())}].");
        return features;
    }

    /// <summary>A [tokens, width] embedding as one [1, tokens, width] context; rank-3 contexts pass through.</summary>
    private static Tensor<T> AsTokens(Tensor<T> embedding)
        => embedding.Rank == 2
            ? AiDotNetEngine.Current.Reshape(embedding, new[] { 1, embedding.Shape[0], embedding.Shape[1] })
            : embedding;

    private static Tensor<T>?[] RepeatCondition(Tensor<T>? condition, int frames)
    {
        var repeated = new Tensor<T>?[frames];
        for (int v = 0; v < frames; v++) repeated[v] = condition;
        return repeated;
    }

    /// <summary>The first <paramref name="views"/> frames of the latents followed by <paramref name="frame"/>.</summary>
    private static Tensor<T> WithLastFrame(IEngine engine, Tensor<T> latents, int views, Tensor<T> frame)
        => engine.TensorConcatenate(new[] { engine.TensorNarrow(latents, 0, 0, views).Contiguous(), frame }, 0);

    /// <summary>
    /// Computes Score Distillation Sampling gradients for 3D optimization.
    /// </summary>
    /// <param name="renderedViews">Rendered views from 3D representation.</param>
    /// <param name="prompt">Text description guiding the 3D.</param>
    /// <param name="timestep">Diffusion timestep for noise level.</param>
    /// <param name="guidanceScale">CFG scale (typically 100 for SDS).</param>
    /// <returns>Gradient tensor for backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Score Distillation uses the diffusion model's
    /// knowledge to guide 3D optimization:
    ///
    /// 1. Render your 3D model from multiple angles
    /// 2. Add noise to the renders
    /// 3. Ask the diffusion model what noise it "sees"
    /// 4. The difference tells you how to improve the 3D model
    ///
    /// High guidance scale (100) strongly pushes toward the text description.
    /// </para>
    /// </remarks>
    public override Tensor<T> ComputeScoreDistillationGradients(
        Tensor<T> renderedViews,
        string prompt,
        int timestep,
        double guidanceScale = 100.0)
        => ComputeScoreDistillationGradients(renderedViews, prompt, timestep, guidanceScale,
            _config.DefaultElevation, _config.DefaultAzimuthStart);

    /// <summary>
    /// Computes MVDream's multi-view Score Distillation Sampling gradients for views rendered from
    /// <c>get_camera(views, elevation, azimuthStart)</c> cameras.
    /// </summary>
    /// <param name="renderedViews">The rendered views [views, 3, H, W], view v at azimuth
    /// <c>azimuthStart + v · AzimuthSpan / views</c>.</param>
    /// <param name="prompt">Text description guiding the 3D.</param>
    /// <param name="timestep">Diffusion timestep for the noise level.</param>
    /// <param name="guidanceScale">CFG scale.</param>
    /// <param name="elevation">The rendering cameras' elevation in degrees (positive looks down).</param>
    /// <param name="azimuthStart">The first rendering camera's azimuth in degrees.</param>
    /// <returns>The gradient with respect to the views' latents.</returns>
    /// <remarks>The camera embedding must describe the cameras the views were rendered from;
    /// <see cref="SampleScoreDistillationCamera"/> draws them the way MVDream's SDS does.</remarks>
    public Tensor<T> ComputeScoreDistillationGradients(
        Tensor<T> renderedViews,
        string prompt,
        int timestep,
        double guidanceScale,
        double elevation,
        double azimuthStart)
    {
        if (renderedViews is null) throw new ArgumentNullException(nameof(renderedViews));
        if (_textConditioner == null)
            throw new InvalidOperationException("Score distillation requires a text conditioning module.");
        RequireTextOnlyUNet("Text score distillation");

        var numViews = renderedViews.Shape[0];

        var promptEmbedding = _textConditioner.EncodeText(_textConditioner.Tokenize(prompt));
        var uncondEmbedding = _textConditioner.GetUnconditionalEmbedding(1);

        // Encode rendered views to latent
        var latents = new Tensor<T>(new[] { numViews, MVDREAM_LATENT_CHANNELS,
            renderedViews.Shape[2] / 8, renderedViews.Shape[3] / 8 });

        for (int v = 0; v < numViews; v++)
        {
            var view = ExtractViewFromBatch(renderedViews, v);
            var viewLatent = _imageVAE.Encode(view, sampleMode: false);
            viewLatent = _imageVAE.ScaleLatent(viewLatent);
            SetView(latents, v, viewLatent);
        }

        // Add noise at specified timestep
        var noise = SampleNoiseTensor(latents._shape, RandomGenerator);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timestep);
        var noisyLatentsTensor = new Tensor<T>(latents._shape, noisyLatents);

        var cameras = CameraEmbedding<T>.CameraTensor(
            CameraEmbedding<T>.GetCameras(numViews, elevation, azimuthStart, _config.AzimuthSpan));

        var condNoisePred = _multiViewUNet.PredictNoiseMultiView(
            noisyLatentsTensor, timestep, RepeatCondition(promptEmbedding, numViews), cameras);
        var uncondNoisePred = _multiViewUNet.PredictNoiseMultiView(
            noisyLatentsTensor, timestep, RepeatCondition(uncondEmbedding, numViews), cameras);

        return ScoreDistillationGradient(condNoisePred, uncondNoisePred, noise, guidanceScale,
            Scheduler.GetAlphaCumulativeProduct(timestep));
    }

    /// <summary>
    /// Draws a score-distillation camera set the way MVDream's SDS does: an elevation uniform in
    /// [<see cref="MVDreamConfig.ScoreDistillationMinElevation"/>, <see cref="MVDreamConfig.ScoreDistillationMaxElevation"/>]
    /// degrees and a first azimuth uniform in [0, 360) degrees; the views follow it <c>AzimuthSpan / views</c> apart.
    /// </summary>
    /// <param name="seed">Optional seed; a secure draw when null.</param>
    /// <returns>The elevation and the first azimuth, in degrees.</returns>
    public (double Elevation, double AzimuthStart) SampleScoreDistillationCamera(int? seed = null)
    {
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        double low = _config.ScoreDistillationMinElevation, high = _config.ScoreDistillationMaxElevation;
        if (high < low)
            throw new InvalidOperationException("ScoreDistillationMaxElevation must not be below ScoreDistillationMinElevation.");
        double elevation = low + ((high - low) * rng.NextDouble());
        double azimuth = 360.0 * rng.NextDouble();
        return (elevation, azimuth);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// The SDS gradient of DreamFusion (Poole et al., 2022, eq. 3) used by MVDream's multi-view SDS:
    /// <c>w(t) (eps_cfg - eps)</c> with <c>eps_cfg = eps_uncond + s (eps_cond - eps_uncond)</c> and
    /// <c>w(t) = sigma_t^2 = 1 - alpha_bar_t</c>.
    /// </summary>
    internal static Tensor<T> ScoreDistillationGradient(
        Tensor<T> conditional, Tensor<T> unconditional, Tensor<T> noise, double guidanceScale, T alphaCumprod)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var guided = engine.TensorAdd(unconditional,
            engine.TensorMultiplyScalar(engine.TensorSubtract(conditional, unconditional), ops.FromDouble(guidanceScale)));
        return engine.TensorMultiplyScalar(engine.TensorSubtract(guided, noise), ops.Subtract(ops.One, alphaCumprod));
    }

    /// <summary>
    /// Generates uniformly distributed camera positions around the object.
    /// </summary>
    private (double azimuth, double elevation, double radius)[] GenerateCameraPositions(
        int numViews, double elevation)
    {
        var positions = new (double, double, double)[numViews];
        var elevationRad = elevation * Math.PI / 180.0;

        for (int v = 0; v < numViews; v++)
        {
            var azimuth = 2.0 * Math.PI * v / numViews;
            positions[v] = (azimuth, elevationRad, DEFAULT_CAMERA_DISTANCE);
        }

        return positions;
    }

    /// <summary>
    /// Extracts a single view from multi-view tensor.
    /// </summary>
    private Tensor<T> ExtractView(Tensor<T> multiView, int viewIndex)
    {
        var viewShape = new[] { 1, multiView.Shape[1], multiView.Shape[2], multiView.Shape[3] };
        var view = new Tensor<T>(viewShape);
        var viewSpan = view.AsWritableSpan();
        var multiSpan = multiView.AsSpan();

        var viewSize = multiView.Shape[1] * multiView.Shape[2] * multiView.Shape[3];
        var viewOffset = viewIndex * viewSize;

        int count = Math.Max(0, Math.Min(viewSize, multiSpan.Length - viewOffset));
        multiSpan.Slice(viewOffset, count).CopyTo(viewSpan);

        return view;
    }

    /// <summary>
    /// Extracts a single view from a batch tensor with batch dimension.
    /// </summary>
    private Tensor<T> ExtractViewFromBatch(Tensor<T> batch, int viewIndex)
    {
        return ExtractView(batch, viewIndex);
    }

    /// <summary>
    /// Sets a single view in multi-view tensor.
    /// </summary>
    private void SetView(Tensor<T> multiView, int viewIndex, Tensor<T> view)
    {
        var multiSpan = multiView.AsWritableSpan();
        var viewSpan = view.AsSpan();

        var viewSize = multiView.Shape[1] * multiView.Shape[2] * multiView.Shape[3];
        var viewOffset = viewIndex * viewSize;

        int count = Math.Max(0, Math.Min(Math.Min(viewSize, viewSpan.Length), multiSpan.Length - viewOffset));
        viewSpan.Slice(0, count).CopyTo(multiSpan.Slice(viewOffset, count));
    }

    /// <summary>
    /// Reconstructs mesh from multiple views using depth-based back-projection and visual hull carving.
    /// </summary>
    /// <remarks>
    /// This implementation uses:
    /// 1. Depth estimation from each view using intensity gradients and defocus cues
    /// 2. Back-projection of depth maps to 3D points using camera parameters
    /// 3. Visual hull carving to refine the point cloud
    /// 4. Point cloud filtering and merging across views
    /// </remarks>
    private Mesh3D<T> ReconstructFromMultiView(
        Tensor<T>[] views,
        (double azimuth, double elevation, double radius)[] cameras)
    {
        if (views.Length == 0 || cameras.Length == 0)
        {
            throw new ArgumentException("At least one view and camera are required for reconstruction.");
        }

        var allPoints = new List<(double x, double y, double z, double confidence)>();
        int imageHeight = views[0].Shape[^2];
        int imageWidth = views[0].Shape[^1];

        // Process each view
        for (int v = 0; v < views.Length; v++)
        {
            var view = views[v];
            var cam = cameras[v];

            // Step 1: Estimate depth from this view using gradient-based depth cues
            var depthMap = EstimateDepthFromView(view, imageHeight, imageWidth);

            // Step 2: Extract silhouette mask (non-background pixels)
            var silhouetteMask = ExtractSilhouette(view, imageHeight, imageWidth);

            // Grayscale intensity (mean of the colour channels) drives the per-point confidence.
            var intensity = GrayscaleIntensity(view, imageHeight, imageWidth).Cast<double>().ToArray();

            for (int y = 0; y < imageHeight; y++)
            {
                for (int x = 0; x < imageWidth; x++)
                {
                    if (!silhouetteMask[y, x])
                    {
                        continue;
                    }

                    double depth = depthMap[y, x];
                    if (depth <= 0.01 || depth >= 10.0)
                    {
                        continue;
                    }

                    // Compute normalized image coordinates [-1, 1]
                    double nx = (2.0 * x / imageWidth) - 1.0;
                    double ny = 1.0 - (2.0 * y / imageHeight);

                    // Back-project to camera space
                    double camX = nx * depth * 0.5; // Assuming ~90 degree FOV
                    double camY = ny * depth * 0.5;
                    double camZ = -depth;

                    // Transform to world space using inverse camera rotation
                    var (worldX, worldY, worldZ) = TransformCameraToWorld(
                        camX, camY, camZ,
                        cam.azimuth, cam.elevation, cam.radius);

                    double confidence = Math.Min(1.0, Math.Max(0.1, intensity[(y * imageWidth) + x]));
                    allPoints.Add((worldX, worldY, worldZ, confidence));
                }
            }
        }

        // Step 4: Filter and downsample point cloud
        var filteredPoints = FilterPointCloud(allPoints, targetCount: DefaultPointCount);

        // Step 5: Convert to tensor (one bulk conversion of the coordinates to T)
        var coordinates = new double[filteredPoints.Count * 3];
        for (int i = 0; i < filteredPoints.Count; i++)
        {
            var p = filteredPoints[i];
            coordinates[i * 3] = p.x;
            coordinates[(i * 3) + 1] = p.y;
            coordinates[(i * 3) + 2] = p.z;
        }

        var points = new Tensor<double>(new[] { 1, filteredPoints.Count, 3 }, new Vector<double>(coordinates)).Cast<T>();
        return PointCloudToMesh(points, SurfaceReconstructionMethod.Poisson);
    }

    /// <summary>
    /// The grayscale intensity [height, width] of the first image of <paramref name="view"/>: the mean of its
    /// first min(3, channels) channels.
    /// </summary>
    internal Tensor<T> GrayscaleIntensity(Tensor<T> view, int height, int width)
    {
        int channels = view.Shape[^3];
        int used = Math.Min(3, channels);
        var images = view.Reshape(new[] { view.Length / (channels * height * width), channels, height, width });
        var colour = Engine.TensorNarrow(Engine.TensorNarrow(images, 0, 0, 1), 1, 0, used);
        return Engine.ReduceMean(colour, new[] { 0, 1 }, false).Reshape(new[] { height, width });
    }

    /// <summary>
    /// Estimates depth from a single view using gradient-based cues.
    /// Uses a combination of Laplacian-based focus measure and intensity gradients.
    /// </summary>
    /// <remarks>
    /// On the interior pixels: sharpness = min(1, 4 * (|laplacian| + 0.5 * |sobel gradient|) / 2) and
    /// depth = 0.5 + 1.5 * (1 - sharpness) (sharper = closer); the border copies its nearest interior value.
    /// </remarks>
    internal double[,] EstimateDepthFromView(Tensor<T> view, int height, int width)
    {
        var depthMap = new double[height, width];
        if (height < 3 || width < 3)
        {
            // No interior: a flat (fully defocused) depth of 2.
            var flat = Enumerable.Repeat(2.0, height * width).ToArray();
            Buffer.BlockCopy(flat, 0, depthMap, 0, flat.Length * sizeof(double));
            return depthMap;
        }

        var intensity = GrayscaleIntensity(view, height, width);
        Tensor<T> Shifted(int dy, int dx) =>
            Engine.TensorNarrow(Engine.TensorNarrow(intensity, 0, 1 + dy, height - 2), 1, 1 + dx, width - 2);

        var centre = Shifted(0, 0);
        var laplacian = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorAdd(Shifted(-1, 0), Shifted(1, 0)), Engine.TensorAdd(Shifted(0, -1), Shifted(0, 1))),
            Engine.TensorMultiplyScalar(centre, NumOps.FromDouble(4.0)));
        var two = NumOps.FromDouble(2.0);
        var gx = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorAdd(Shifted(-1, 1), Engine.TensorMultiplyScalar(Shifted(0, 1), two)), Shifted(1, 1)),
            Engine.TensorAdd(Engine.TensorAdd(Shifted(-1, -1), Engine.TensorMultiplyScalar(Shifted(0, -1), two)), Shifted(1, -1)));
        var gy = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorAdd(Shifted(1, -1), Engine.TensorMultiplyScalar(Shifted(1, 0), two)), Shifted(1, 1)),
            Engine.TensorAdd(Engine.TensorAdd(Shifted(-1, -1), Engine.TensorMultiplyScalar(Shifted(-1, 0), two)), Shifted(-1, 1)));
        var gradient = Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(gx), Engine.TensorSquare(gy)));

        // (|laplacian| + 0.5 * gradient) / 2 * 4 = 2 |laplacian| + gradient, capped at 1.
        var sharpness = Engine.TensorClamp(
            Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.TensorAbs(laplacian), two), gradient),
            NumOps.Zero, NumOps.One);
        var interior = Engine.ScalarMinusTensor(NumOps.FromDouble(2.0), Engine.TensorMultiplyScalar(sharpness, NumOps.FromDouble(1.5)));

        // Fill borders by replicating the nearest interior column, then row.
        var columns = Engine.TensorConcatenate(new[]
        {
            Engine.TensorNarrow(interior, 1, 0, 1), interior, Engine.TensorNarrow(interior, 1, width - 3, 1)
        }, 1);
        var full = Engine.TensorConcatenate(new[]
        {
            Engine.TensorNarrow(columns, 0, 0, 1), columns, Engine.TensorNarrow(columns, 0, height - 3, 1)
        }, 0);

        var values = full.Cast<double>().ToArray();
        Buffer.BlockCopy(values, 0, depthMap, 0, values.Length * sizeof(double));
        return depthMap;
    }

    /// <summary>
    /// Extracts a silhouette mask identifying foreground pixels: those whose grayscale intensity differs by more
    /// than 0.15 from the background level estimated on the top-left corner block.
    /// </summary>
    internal bool[,] ExtractSilhouette(Tensor<T> view, int height, int width)
    {
        var mask = new bool[height, width];
        var intensity = GrayscaleIntensity(view, height, width);

        // Estimate background color from the corner block
        int cornerSize = Math.Min(Math.Max(4, Math.Min(height, width) / 16), Math.Min(height, width));
        T background = Engine.TensorMean(
            Engine.TensorNarrow(Engine.TensorNarrow(intensity, 0, 0, cornerSize), 1, 0, cornerSize));

        // Mark as foreground if significantly different from background
        var foreground = Engine.TensorGreaterThan(
            Engine.TensorAbs(Engine.TensorSubtractScalar(intensity, background)), NumOps.FromDouble(0.15))
            .Cast<int>().ToArray();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                mask[y, x] = foreground[(y * width) + x] != 0;
        }

        return mask;
    }

    /// <summary>
    /// Builds a camera matrix from spherical coordinates.
    /// </summary>
    private double[,] BuildCameraMatrix(double azimuth, double elevation, double radius)
    {
        // Convert spherical to Cartesian camera position
        double camX = radius * Math.Cos(elevation) * Math.Sin(azimuth);
        double camY = radius * Math.Sin(elevation);
        double camZ = radius * Math.Cos(elevation) * Math.Cos(azimuth);

        // Camera looks at origin
        double lookX = -camX, lookY = -camY, lookZ = -camZ;
        double lookLen = Math.Sqrt(lookX * lookX + lookY * lookY + lookZ * lookZ);
        lookX /= lookLen; lookY /= lookLen; lookZ /= lookLen;

        // Up vector (world Y)
        double upX = 0, upY = 1, upZ = 0;

        // Right vector
        double rightX = upY * lookZ - upZ * lookY;
        double rightY = upZ * lookX - upX * lookZ;
        double rightZ = upX * lookY - upY * lookX;
        double rightLen = Math.Sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
        if (rightLen > 1e-6)
        {
            rightX /= rightLen; rightY /= rightLen; rightZ /= rightLen;
        }

        // Recompute up
        upX = lookY * rightZ - lookZ * rightY;
        upY = lookZ * rightX - lookX * rightZ;
        upZ = lookX * rightY - lookY * rightX;

        return new double[,]
        {
            { rightX, upX, -lookX, camX },
            { rightY, upY, -lookY, camY },
            { rightZ, upZ, -lookZ, camZ },
            { 0, 0, 0, 1 }
        };
    }

    /// <summary>
    /// Transforms a point from camera space (looking down -Z, +Y up) to world space with the same
    /// camera-to-world matrix the camera embedding conditions generation on
    /// (<see cref="CameraEmbedding{T}.CameraToWorld"/>), so reconstruction and conditioning share one basis.
    /// </summary>
    internal static (double x, double y, double z) TransformCameraToWorld(
        double camX, double camY, double camZ,
        double azimuth, double elevation, double radius)
    {
        // Row-major [right | up | back | position]: world = right * x + up * y + back * z + position.
        var m = CameraEmbedding<T>.CameraToWorld(azimuth, elevation, radius);
        double worldX = (m[0] * camX) + (m[1] * camY) + (m[2] * camZ) + m[3];
        double worldY = (m[4] * camX) + (m[5] * camY) + (m[6] * camZ) + m[7];
        double worldZ = (m[8] * camX) + (m[9] * camY) + (m[10] * camZ) + m[11];
        return (worldX, worldY, worldZ);
    }

    /// <summary>
    /// Filters and downsamples a point cloud using voxel grid filtering.
    /// </summary>
    private List<(double x, double y, double z)> FilterPointCloud(
        List<(double x, double y, double z, double confidence)> points,
        int targetCount)
    {
        if (points.Count == 0)
        {
            throw new InvalidOperationException(
                "Point cloud reconstruction failed: no valid 3D points were extracted from the multi-view images. " +
                "This may indicate failures in depth estimation or silhouette extraction.");
        }

        // Find bounds
        double minX = double.MaxValue, maxX = double.MinValue;
        double minY = double.MaxValue, maxY = double.MinValue;
        double minZ = double.MaxValue, maxZ = double.MinValue;

        foreach (var p in points)
        {
            minX = Math.Min(minX, p.x); maxX = Math.Max(maxX, p.x);
            minY = Math.Min(minY, p.y); maxY = Math.Max(maxY, p.y);
            minZ = Math.Min(minZ, p.z); maxZ = Math.Max(maxZ, p.z);
        }

        // Voxel grid filtering
        double extent = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
        if (extent < 1e-6) extent = 1.0;

        int gridRes = (int)Math.Ceiling(Math.Pow(targetCount, 1.0 / 3.0) * 2);
        double voxelSize = extent / gridRes;

        var voxelGrid = new Dictionary<(int, int, int), (double x, double y, double z, double weight, int count)>();

        foreach (var p in points)
        {
            int vx = (int)((p.x - minX) / voxelSize);
            int vy = (int)((p.y - minY) / voxelSize);
            int vz = (int)((p.z - minZ) / voxelSize);
            var key = (vx, vy, vz);

            if (voxelGrid.TryGetValue(key, out var existing))
            {
                // Weighted average based on confidence
                double newWeight = existing.weight + p.confidence;
                double nx = (existing.x * existing.weight + p.x * p.confidence) / newWeight;
                double ny = (existing.y * existing.weight + p.y * p.confidence) / newWeight;
                double nz = (existing.z * existing.weight + p.z * p.confidence) / newWeight;
                voxelGrid[key] = (nx, ny, nz, newWeight, existing.count + 1);
            }
            else
            {
                voxelGrid[key] = (p.x, p.y, p.z, p.confidence, 1);
            }
        }

        // Convert to list and sample if needed
        var filtered = voxelGrid.Values
            .OrderByDescending(v => v.count)
            .Select(v => (v.x, v.y, v.z))
            .ToList();

        if (filtered.Count > targetCount)
        {
            // Subsample uniformly
            var step = (double)filtered.Count / targetCount;
            var subsampled = new List<(double, double, double)>();
            for (int i = 0; i < targetCount; i++)
            {
                int idx = (int)(i * step);
                if (idx < filtered.Count)
                {
                    subsampled.Add(filtered[idx]);
                }
            }
            return subsampled;
        }

        // If too few points, add some by interpolation
        while (filtered.Count < targetCount && filtered.Count > 0)
        {
            var rng = RandomGenerator;
            int idx1 = rng.Next(filtered.Count);
            int idx2 = rng.Next(filtered.Count);
            var p1 = filtered[idx1];
            var p2 = filtered[idx2];
            double t = rng.NextDouble();
            filtered.Add((
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y),
                p1.z + t * (p2.z - p1.z)
            ));
        }

        return filtered;
    }

    #endregion

    #region IParameterizable Implementation





    /// <inheritdoc />
    #endregion

    #region ICloneable Implementation

    #endregion
}

/// <summary>
/// Configuration for MVDream model.
/// </summary>
public class MVDreamConfig
{
    /// <summary>
    /// Default number of views to generate.
    /// </summary>
    public int DefaultNumViews { get; set; } = 4;

    /// <summary>
    /// Image resolution for each view.
    /// </summary>
    public int ImageSize { get; set; } = 256;

    /// <summary>
    /// Informational: whether the model's U-Net attends across views. The model does not read it; the 3D
    /// self-attention is part of the multi-view U-Net's structure (<see cref="MultiViewUNetOptions"/>), fixed when the
    /// U-Net is built.
    /// </summary>
    public bool UseMultiViewAttention { get; set; } = true;

    /// <summary>
    /// Camera distance from object center.
    /// </summary>
    public double CameraDistance { get; set; } = 1.5;

    /// <summary>
    /// The text-to-multi-view cameras' elevation in degrees, positive looking down (15: <c>--camera_elev</c> of the
    /// official <c>scripts/t2i.py</c>).
    /// </summary>
    public double DefaultElevation { get; set; } = 15.0;

    /// <summary>
    /// The text-to-multi-view first view's azimuth in degrees (90: <c>--camera_azim</c> of <c>scripts/t2i.py</c>).
    /// </summary>
    public double DefaultAzimuthStart { get; set; } = 90.0;

    /// <summary>The azimuth range the views spread over, in degrees (360: <c>--camera_azim_span</c>).</summary>
    public double AzimuthSpan { get; set; } = 360.0;

    /// <summary>
    /// The image-to-multi-view cameras' elevation in degrees (5: ImageDream's <c>scripts/demo.py</c>,
    /// <c>get_camera(4, elevation=5, azimuth_start=0, extra_view=True)</c>). The image path's first view is at
    /// azimuth 0, the input's own viewpoint.
    /// </summary>
    public double ImageElevation { get; set; } = 5.0;

    /// <summary>
    /// The lowest elevation, in degrees, the score-distillation camera sampler draws (0). MVDream's SDS samples the
    /// elevation uniformly in [0, 30] degrees and a random azimuth for the first of four views 90 degrees apart (Shi et
    /// al., 2023, Sec. 4 and the released threestudio configuration).
    /// </summary>
    public double ScoreDistillationMinElevation { get; set; } = 0.0;

    /// <summary>The highest elevation, in degrees, the score-distillation camera sampler draws (30).</summary>
    public double ScoreDistillationMaxElevation { get; set; } = 30.0;

    /// <summary>
    /// Gets the default configuration.
    /// </summary>
    public static MVDreamConfig Default => new MVDreamConfig();
}

/// <summary>
/// MVDream's multi-view U-Net (<c>MultiViewUNetModel</c>, Shi et al., 2023; and ImageDream's, Wang and Shi, 2023): the
/// shared <see cref="UNetNoisePredictor{T}"/> built with <see cref="MultiViewUNetOptions"/> (3D self-attention in every
/// transformer), the camera embedding added to its time embedding, and, for ImageDream, the image-prompt resampler.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para>
/// The official sd-v2-base configuration (<c>mvdream/configs/sd-v2-base.yaml</c>): model_channels 320, channel_mult
/// [1, 2, 4, 4], num_res_blocks 2, attention at the first three levels (attention_resolutions [4, 2, 1] are
/// downsampling factors 1, 2 and 4), num_head_channels 64, one linear-projection transformer per position, context_dim
/// 1024, camera_dim 16. All the views of one object run through the U-Net as one batch, so the 3D self-attention sees
/// all their tokens.
/// </para>
/// <para>
/// Camera conditioning is <c>emb = time_embed(t) + camera_embed(camera)</c> with
/// <c>camera_embed = Linear(16, 4·320), SiLU, Linear(4·320, 4·320)</c> (<see cref="CameraEmbedding"/>).
/// </para>
/// </remarks>
public class MultiViewUNet<T> : IParameterSource<T>, ICloneable<MultiViewUNet<T>>
{
    private UNetNoisePredictor<T> _baseUNet;
    private readonly CameraEmbedding<T> _cameraEmbedding;
    private readonly ImageDreamResampler<T>? _imageEmbed;
    private readonly int _numViews;
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _baseChannels;
    private readonly int _contextDim;
    private readonly MultiViewUNetOptions _options;

    /// <summary>
    /// Gets whether classifier-free guidance is supported.
    /// </summary>
    public bool SupportsCFG => true;

    /// <summary>
    /// Gets the base UNet: the shared U-Net built with the multi-view option.
    /// </summary>
    public UNetNoisePredictor<T> BaseUNet => _baseUNet;

    /// <summary>
    /// Gets the camera embedding (<c>camera_embed</c>) whose output is added to the U-Net's time embedding.
    /// </summary>
    public CameraEmbedding<T> CameraEmbedding => _cameraEmbedding;

    /// <summary>The multi-view options the base U-Net was built with.</summary>
    public MultiViewUNetOptions Options => _options;

    /// <summary>The number of views generated per object by default.</summary>
    public int NumViews => _numViews;

    /// <summary>Whether this U-Net has ImageDream's image-prompt resampler and decoupled image cross-attention.</summary>
    public bool HasImagePrompt => _imageEmbed is not null;

    /// <summary>ImageDream's resampler (<c>image_embed</c>), or null for the text-only MVDream U-Net.</summary>
    internal ImageDreamResampler<T>? ImageEmbed => _imageEmbed;

    /// <summary>
    /// Creates a multi-view U-Net.
    /// </summary>
    /// <param name="inputChannels">Latent channels in (4).</param>
    /// <param name="outputChannels">Predicted channels (4).</param>
    /// <param name="baseChannels">The U-Net's model_channels (320).</param>
    /// <param name="numViews">Views generated per object by default (4).</param>
    /// <param name="contextDim">The cross-attention context width (1024, OpenCLIP ViT-H's penultimate text states).</param>
    /// <param name="seed">Optional seed for the initial weights.</param>
    /// <param name="options">The multi-view options; <see cref="MultiViewUNetOptions.MVDream"/> when null. With
    /// <see cref="MultiViewUNetOptions.ImagePromptTokens"/> above zero (and a context) the U-Net is ImageDream's and
    /// carries the resampler.</param>
    public MultiViewUNet(
        int inputChannels,
        int outputChannels,
        int baseChannels,
        int numViews,
        int contextDim,
        int? seed = null,
        MultiViewUNetOptions? options = null)
    {
        if (numViews <= 0) throw new ArgumentOutOfRangeException(nameof(numViews), "At least one view is required.");
        _options = options ?? MultiViewUNetOptions.MVDream();
        _options.Validate();
        _numViews = numViews;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _baseChannels = baseChannels;
        _contextDim = contextDim;
        _baseUNet = CreateBaseUNet(seed);
        _cameraEmbedding = new CameraEmbedding<T>(embeddingDim: _baseUNet.TimeEmbeddingDim, seed: seed);
        if (_options.ImagePromptTokens > 0 && contextDim > 0)
        {
            _imageEmbed = new ImageDreamResampler<T>(
                dim: contextDim,
                depth: _options.ResamplerDepth,
                headDim: _options.ResamplerHeadDim,
                heads: _options.ResamplerHeads,
                queries: _options.ImagePromptTokens,
                embeddingDim: _options.ImageFeatureDim,
                outputDim: contextDim,
                feedForwardMultiplier: _options.ResamplerFeedForwardMultiplier);
        }
    }

    private UNetNoisePredictor<T> CreateBaseUNet(int? seed) => new UNetNoisePredictor<T>(
        inputChannels: _inputChannels,
        outputChannels: _outputChannels,
        baseChannels: _baseChannels,
        channelMultipliers: new[] { 1, 2, 4, 4 },
        numResBlocks: 2,
        // LDM's attention_resolutions [4, 2, 1] are downsampling factors: levels 0, 1 and 2 attend, level 3 does not.
        attentionResolutions: new[] { 0, 1, 2 },
        contextDim: _contextDim,
        inputHeight: MVDreamModel<T>.MVDREAM_IMAGE_SIZE / 8,
        seed: seed,
        multiView: _options);

    /// <inheritdoc cref="Clone"/>
    public MultiViewUNet<T> DeepCopy() => Clone();

    /// <summary>
    /// Deep-copies this multi-view U-Net, preserving trained weights: an identically configured instance whose base
    /// U-Net receives this one's parameter chunks and whose camera embedding and resampler receive this one's values.
    /// </summary>
    // Declaring ICloneable is what lets CloneEngine find this Clone. Without it MVDreamModel's configuration copy
    // received the SOURCE's multi-view U-Net as its constructor argument, so the clone and the original shared every
    // base U-Net layer and Clone threw AliasedLayerGraph (#2087). The base U-Net is rebuilt with the same multi-view
    // options and its weights copied by chunk: two lazy predictors each draw their own weights the first time they are
    // read, so a structural copy alone would silently be a different model.
    public MultiViewUNet<T> Clone()
    {
        var clone = new MultiViewUNet<T>(_inputChannels, _outputChannels, _baseChannels, _numViews, _contextDim,
            seed: null, options: CopyOptions(_options));
        clone._baseUNet.SetParameterChunks(_baseUNet.GetParameterChunks());
        clone._cameraEmbedding.SetParameters(_cameraEmbedding.GetParameters());
        if (_imageEmbed is not null && clone._imageEmbed is not null)
            clone._imageEmbed.SetParameters(_imageEmbed.GetParameters());
        return clone;
    }

    private static MultiViewUNetOptions CopyOptions(MultiViewUNetOptions o) => new MultiViewUNetOptions
    {
        FramesPerObject = o.FramesPerObject,
        HeadChannels = o.HeadChannels,
        TransformerDepth = o.TransformerDepth,
        ImagePromptTokens = o.ImagePromptTokens,
        ImagePromptWeight = o.ImagePromptWeight,
        ImageFeatureDim = o.ImageFeatureDim,
        ResamplerDepth = o.ResamplerDepth,
        ResamplerHeads = o.ResamplerHeads,
        ResamplerHeadDim = o.ResamplerHeadDim,
        ResamplerFeedForwardMultiplier = o.ResamplerFeedForwardMultiplier
    };

    /// <summary>
    /// Predicts noise for a single view.
    /// </summary>
    public Tensor<T> PredictNoise(Tensor<T> input, int timestep, Tensor<T>? conditioning = null)
    {
        return _baseUNet.PredictNoise(input, timestep, conditioning);
    }

    /// <summary>
    /// Predicts noise for the views of one object together, without camera conditioning (MVDream's U-Net with
    /// <c>camera=None</c>): the views run as one batch, so the 3D self-attention attends across them.
    /// </summary>
    /// <param name="multiViewInput">The view latents [views, channels, height, width].</param>
    /// <param name="timestep">The timestep.</param>
    /// <param name="viewConditions">One cross-attention context per view.</param>
    public Tensor<T> PredictNoiseMultiView(Tensor<T> multiViewInput, int timestep, Tensor<T>[] viewConditions)
    {
        if (multiViewInput is null) throw new ArgumentNullException(nameof(multiViewInput));
        if (viewConditions is null) throw new ArgumentNullException(nameof(viewConditions));
        var context = StackConditions(viewConditions, multiViewInput.Shape[0]);
        return _baseUNet.PredictNoise(multiViewInput, timestep, context);
    }

    /// <summary>
    /// Predicts noise for the views of one object together, adding each view's camera embedding to the U-Net's time
    /// embedding (MVDream's <c>emb = time_embed(t) + camera_embed(camera)</c>).
    /// </summary>
    /// <param name="multiViewInput">The view latents [views, channels, height, width].</param>
    /// <param name="timestep">The timestep.</param>
    /// <param name="viewConditions">Per-view cross-attention conditioning.</param>
    /// <param name="cameras">The flattened 4x4 camera matrices [views, 16].</param>
    internal Tensor<T> PredictNoiseMultiView(
        Tensor<T> multiViewInput, int timestep, Tensor<T>?[] viewConditions, Tensor<T> cameras)
    {
        if (multiViewInput is null) throw new ArgumentNullException(nameof(multiViewInput));
        if (cameras is null) throw new ArgumentNullException(nameof(cameras));
        int numViews = multiViewInput.Shape[0];
        if (cameras.Rank != 2 || cameras.Shape[0] != numViews || cameras.Shape[1] != CameraEmbedding<T>.CameraDim)
            throw new ArgumentException(
                $"Expected [{numViews}, {CameraEmbedding<T>.CameraDim}] cameras; got [{string.Join(", ", cameras.Shape.ToArray())}].",
                nameof(cameras));
        var context = StackConditions(viewConditions, numViews);
        return _baseUNet.PredictNoiseWithAddedTimeEmbedding(multiViewInput, timestep, context, _cameraEmbedding.EmbedCameras(cameras));
    }

    /// <summary>
    /// ImageDream's <c>image_embed</c>: resamples CLIP image features [B, N, ImageFeatureDim] to the image-prompt tokens
    /// [B, ImagePromptTokens, contextDim] the decoupled cross-attention reads after the text tokens.
    /// </summary>
    internal Tensor<T> EmbedImagePrompt(Tensor<T> imageFeatures)
    {
        if (_imageEmbed is null)
            throw new InvalidOperationException(
                "This multi-view U-Net has no image-prompt resampler; build it with MultiViewUNetOptions.ImageDream().");
        return _imageEmbed.Forward(imageFeatures);
    }

    // One [views, L, D] context from the per-view contexts, or null when no view has one.
    private static Tensor<T>? StackConditions(Tensor<T>?[]? viewConditions, int numViews)
    {
        if (viewConditions is null) return null;
        if (viewConditions.Length != numViews)
            throw new ArgumentException($"Expected {numViews} view conditions but received {viewConditions.Length}.", nameof(viewConditions));
        int present = viewConditions.Count(c => c is not null);
        if (present == 0) return null;
        if (present != numViews)
            throw new ArgumentException("Either every view has a cross-attention context or none does.", nameof(viewConditions));
        var parts = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var condition = viewConditions[v] ?? throw new ArgumentException("A view condition is missing.", nameof(viewConditions));
            parts[v] = condition.Rank == 2
                ? AiDotNetEngine.Current.Reshape(condition, new[] { 1, condition.Shape[0], condition.Shape[1] })
                : condition;
            if (parts[v].Rank != 3 || parts[v].Shape[0] != 1)
                throw new ArgumentException("Each view condition must be one [1, tokens, width] context.", nameof(viewConditions));
        }
        return AiDotNetEngine.Current.TensorConcatenate(parts, 0);
    }

    /// <summary>
    /// Loads the official MVDream / ImageDream weights this class owns under their state-dict names: every
    /// <c>SpatialTransformer3D</c> (<c>{prefix}input_blocks.*.1.*</c>, <c>middle_block.1.*</c>, <c>output_blocks.*.1.*</c>),
    /// <c>{prefix}camera_embed.{0,2}</c> and, for ImageDream, <c>{prefix}image_embed.*</c>.
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and expected shape.</param>
    /// <param name="prefix">The U-Net's state-dict prefix (<c>model.diffusion_model.</c>).</param>
    /// <returns>The names read, in order.</returns>
    /// <remarks>
    /// The shared U-Net's residual blocks and convolutions are not covered: its upsampler is a stride-2 transposed
    /// convolution, where LDM's is nearest-neighbour interpolation and a 3x3 convolution, so LDM's
    /// <c>output_blocks.*.conv</c> weights have no counterpart.
    /// </remarks>
    public IReadOnlyCollection<string> LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "model.diffusion_model.")
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        var source = AiDotNet.ModelLoading.TorchTensorSource.From(read);
        var names = new List<string>(_baseUNet.LoadMultiViewTransformerWeights(source, prefix));
        names.AddRange(_cameraEmbedding.LoadTorchWeights(source, prefix + "camera_embed."));
        if (_imageEmbed is not null)
            names.AddRange(KlModule<T>.Load(_imageEmbed.NamedTensors(prefix + "image_embed"), read));
        return names;
    }

    /// <summary>
    /// Gets parameters: the base U-Net's, the camera embedding's, then the resampler's.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var all = Vector<T>.Concatenate(_baseUNet.GetParameters(), _cameraEmbedding.GetParameters());
        return _imageEmbed is null ? all : Vector<T>.Concatenate(all, _imageEmbed.GetParameters());
    }

    /// <summary>
    /// Sets parameters in the <see cref="GetParameters"/> layout.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"The multi-view U-Net has {ParameterCount} parameters; got {parameters.Length}.", nameof(parameters));
        int unetCount = checked((int)_baseUNet.ParameterCount);
        int cameraCount = checked((int)_cameraEmbedding.ParameterCount);
        _baseUNet.SetParameters(parameters.GetSubVector(0, unetCount));
        _cameraEmbedding.SetParameters(parameters.GetSubVector(unetCount, cameraCount));
        if (_imageEmbed is not null)
            _imageEmbed.SetParameters(parameters.GetSubVector(unetCount + cameraCount, checked((int)_imageEmbed.ParameterCount)));
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public long ParameterCount => _baseUNet.ParameterCount + _cameraEmbedding.ParameterCount + (_imageEmbed?.ParameterCount ?? 0);
}

/// <summary>
/// MVDream's camera embedding (<c>camera_embed</c>).
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para>
/// As in MVDream (Shi et al., 2023), the camera is described by its flattened 4x4 camera-to-world extrinsic
/// matrix (16 values) and embedded by a two-layer MLP (linear, SiLU, linear) whose output is added to the U-Net's
/// timestep embedding. The cameras the official sampler conditions on are <see cref="GetCameras"/>
/// (<c>mvdream/camera_utils.py get_camera</c>).
/// </para>
/// </remarks>
public class CameraEmbedding<T> : IParameterSource<T>
{
    /// <summary>
    /// Size of the camera description: the flattened 4x4 extrinsic matrix.
    /// </summary>
    internal const int CameraDim = 16;

    private readonly Matrix<T> _projectionWeights;
    private readonly Vector<T> _projectionBias;
    private readonly Matrix<T> _outputWeights;
    private readonly Vector<T> _outputBias;
    private readonly int _embeddingDim;
    private readonly int _inputDim;

    /// <summary>
    /// Creates camera embedding.
    /// </summary>
    /// <param name="embeddingDim">The embedding width; the U-Net's time-embedding width it is added to.</param>
    /// <param name="seed">Optional random seed.</param>
    public CameraEmbedding(int embeddingDim, int? seed = null)
    {
        _embeddingDim = embeddingDim;
        _inputDim = CameraDim;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        _projectionWeights = UniformMatrix(embeddingDim, _inputDim, Math.Sqrt(2.0 / _inputDim), rng);
        _projectionBias = new Vector<T>(embeddingDim);
        _outputWeights = UniformMatrix(embeddingDim, embeddingDim, Math.Sqrt(2.0 / embeddingDim), rng);
        _outputBias = new Vector<T>(embeddingDim);
    }

    private static Matrix<T> UniformMatrix(int rows, int columns, double scale, Random rng)
    {
        var matrix = new Matrix<T>(rows, columns);
        var draws = new double[rows * columns];
        for (int i = 0; i < draws.Length; i++)
            draws[i] = (rng.NextDouble() * 2 - 1) * scale;
        new Tensor<double>(new[] { draws.Length }, new Vector<double>(draws)).Cast<T>().AsSpan()
            .CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    /// <summary>
    /// The flattened (row-major) 4x4 camera-to-world matrix of a camera on a sphere looking at the origin with
    /// world +Y up: columns right, up, back (-forward) and the camera position, as in the MVDream / Zero123
    /// camera convention.
    /// </summary>
    /// <param name="azimuth">Azimuth in radians.</param>
    /// <param name="elevation">Elevation in radians.</param>
    /// <param name="radius">Distance from the origin.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The radius is zero, negative or not finite (the camera would have no direction to the origin), or an angle
    /// is not finite.
    /// </exception>
    internal static double[] CameraToWorld(double azimuth, double elevation, double radius)
    {
        if (double.IsNaN(radius) || double.IsInfinity(radius) || radius <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(radius), radius, "The camera radius must be positive and finite.");
        if (double.IsNaN(azimuth) || double.IsInfinity(azimuth))
            throw new ArgumentOutOfRangeException(nameof(azimuth), azimuth, "The camera azimuth must be finite.");
        if (double.IsNaN(elevation) || double.IsInfinity(elevation))
            throw new ArgumentOutOfRangeException(nameof(elevation), elevation, "The camera elevation must be finite.");

        double px = radius * Math.Cos(elevation) * Math.Sin(azimuth);
        double py = radius * Math.Sin(elevation);
        double pz = radius * Math.Cos(elevation) * Math.Cos(azimuth);

        double length = Math.Sqrt((px * px) + (py * py) + (pz * pz));
        double fx = -px / length, fy = -py / length, fz = -pz / length;

        // right = forward x up(0, 1, 0); fall back to up = (0, 0, 1) when looking straight up or down.
        double rx = -fz, ry = 0.0, rz = fx;
        double rightLength = Math.Sqrt((rx * rx) + (rz * rz));
        if (rightLength < 1e-9)
        {
            rx = fy; ry = -fx; rz = 0.0;
            rightLength = Math.Sqrt((rx * rx) + (ry * ry));
        }

        rx /= rightLength; ry /= rightLength; rz /= rightLength;

        // up = right x forward.
        double ux = (ry * fz) - (rz * fy);
        double uy = (rz * fx) - (rx * fz);
        double uz = (rx * fy) - (ry * fx);

        return new[]
        {
            rx, ux, -fx, px,
            ry, uy, -fy, py,
            rz, uz, -fz, pz,
            0.0, 0.0, 0.0, 1.0
        };
    }

    /// <summary>
    /// The official sampling cameras (<c>mvdream/camera_utils.py get_camera</c>, which ImageDream's copy extends with
    /// <c>extra_view</c>): <paramref name="frames"/> cameras on the unit sphere at one elevation, at azimuths
    /// <c>azimuthStart + i · azimuthSpan / frames</c> degrees, each <c>create_camera_to_world_matrix</c> converted to
    /// Blender's axes (<c>flip_yz · M</c>) and flattened row-major; then, with <paramref name="extraView"/>, ImageDream's
    /// all-zero camera of the input image's frame.
    /// </summary>
    /// <param name="frames">The number of views.</param>
    /// <param name="elevation">The elevation in degrees (positive looks down on the object).</param>
    /// <param name="azimuthStart">The first view's azimuth in degrees.</param>
    /// <param name="azimuthSpan">The azimuth range the views spread over, in degrees (360).</param>
    /// <param name="extraView">Whether to append ImageDream's zero camera.</param>
    /// <returns>The cameras, [frames (+1), 16].</returns>
    internal static double[][] GetCameras(int frames, double elevation, double azimuthStart, double azimuthSpan = 360.0,
        bool extraView = false)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames), "At least one camera is required.");
        var cameras = new double[frames + (extraView ? 1 : 0)][];
        double gap = azimuthSpan / frames;
        for (int i = 0; i < frames; i++)
            cameras[i] = BlenderCameraToWorld(elevation, azimuthStart + (i * gap));
        if (extraView) cameras[frames] = new double[CameraDim];
        return cameras;
    }

    /// <summary>
    /// <c>convert_opengl_to_blender(create_camera_to_world_matrix(elevation, azimuth))</c>, flattened row-major: the unit
    /// sphere camera of <see cref="CameraToWorld"/> with its rows permuted by <c>flip_yz = [[1,0,0,0],[0,0,-1,0],
    /// [0,1,0,0],[0,0,0,1]]</c> (row 1 becomes minus row 2, row 2 becomes row 1).
    /// </summary>
    internal static double[] BlenderCameraToWorld(double elevationDegrees, double azimuthDegrees)
    {
        var m = CameraToWorld(azimuthDegrees * Math.PI / 180.0, elevationDegrees * Math.PI / 180.0, 1.0);
        return new[]
        {
            m[0], m[1], m[2], m[3],
            -m[8], -m[9], -m[10], -m[11],
            m[4], m[5], m[6], m[7],
            m[12], m[13], m[14], m[15]
        };
    }

    /// <summary>The cameras as a [count, 16] tensor.</summary>
    internal static Tensor<T> CameraTensor(double[][] cameras)
    {
        if (cameras is null) throw new ArgumentNullException(nameof(cameras));
        var flat = new double[cameras.Length * CameraDim];
        for (int i = 0; i < cameras.Length; i++)
        {
            if (cameras[i].Length != CameraDim)
                throw new ArgumentException($"A camera has {CameraDim} values; got {cameras[i].Length}.", nameof(cameras));
            Array.Copy(cameras[i], 0, flat, i * CameraDim, CameraDim);
        }
        return new Tensor<double>(new[] { cameras.Length, CameraDim }, new Vector<double>(flat)).Cast<T>();
    }

    /// <summary>
    /// Embeds a batch of flattened cameras <c>[N, 16]</c> to <c>[N, embeddingDim]</c>:
    /// <c>W_2 SiLU(W_1 c + b_1) + b_2</c> per row (<c>camera_embed(camera)</c>).
    /// </summary>
    internal Tensor<T> EmbedCameras(Tensor<T> cameras)
    {
        if (cameras is null) throw new ArgumentNullException(nameof(cameras));
        if (cameras.Rank != 2 || cameras.Shape[1] != CameraDim)
            throw new ArgumentException($"Cameras must be [N, {CameraDim}].", nameof(cameras));
        var engine = AiDotNetEngine.Current;
        int count = cameras.Shape[0];
        var rowShape = new[] { count, _embeddingDim };
        var hidden = engine.TensorAdd(
            engine.TensorMatMul(cameras, engine.TensorTranspose(Tensor<T>.FromMatrix(_projectionWeights))),
            engine.TensorBroadcastTo(_projectionBias.AsTensor().Reshape(new[] { 1, _embeddingDim }), rowShape));
        hidden = engine.TensorMultiply(hidden, engine.TensorSigmoid(hidden));
        return engine.TensorAdd(
            engine.TensorMatMul(hidden, engine.TensorTranspose(Tensor<T>.FromMatrix(_outputWeights))),
            engine.TensorBroadcastTo(_outputBias.AsTensor().Reshape(new[] { 1, _embeddingDim }), rowShape));
    }

    /// <summary>
    /// Loads <c>{prefix}0.weight [E, 16]</c>, <c>0.bias</c>, <c>2.weight [E, E]</c> and <c>2.bias</c>: the official
    /// <c>camera_embed = nn.Sequential(linear, nn.SiLU(), linear)</c>, whose torch <c>[out, in]</c> layout is this
    /// embedding's own.
    /// </summary>
    internal IReadOnlyList<string> LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        var values = new List<double>();
        values.AddRange(source.Read(prefix + "0.weight", new[] { _embeddingDim, _inputDim }));
        values.AddRange(source.Read(prefix + "0.bias", new[] { _embeddingDim }));
        values.AddRange(source.Read(prefix + "2.weight", new[] { _embeddingDim, _embeddingDim }));
        values.AddRange(source.Read(prefix + "2.bias", new[] { _embeddingDim }));
        SetParameters(new Tensor<double>(new[] { values.Count }, new Vector<double>(values.ToArray())).Cast<T>().ToVector());
        return new[] { prefix + "0.weight", prefix + "0.bias", prefix + "2.weight", prefix + "2.bias" };
    }

    /// <summary>
    /// Embeds a camera position: <c>W_2 SiLU(W_1 vec(E) + b_1) + b_2</c> for its extrinsic matrix E.
    /// </summary>
    /// <returns>The embedding [1, embeddingDim].</returns>
    public Tensor<T> Embed(double azimuth, double elevation, double radius)
        => EmbedExtrinsic(CameraToWorld(azimuth, elevation, radius));

    /// <summary>
    /// Embeds a flattened (row-major) 4x4 camera description directly; ImageDream's extra image view uses the
    /// all-zero camera.
    /// </summary>
    /// <param name="extrinsic">The 16 camera values.</param>
    /// <returns>The embedding [1, embeddingDim].</returns>
    internal Tensor<T> EmbedExtrinsic(double[] extrinsic)
    {
        if (extrinsic is null) throw new ArgumentNullException(nameof(extrinsic));
        if (extrinsic.Length != CameraDim)
            throw new ArgumentException($"A camera has {CameraDim} values; got {extrinsic.Length}.", nameof(extrinsic));
        var engine = AiDotNetEngine.Current;
        var extrinsicTensor = new Tensor<double>(new[] { 1, CameraDim }, new Vector<double>(extrinsic)).Cast<T>();
        var hidden = engine.TensorAdd(
            engine.TensorMatMul(extrinsicTensor, engine.TensorTranspose(Tensor<T>.FromMatrix(_projectionWeights))),
            _projectionBias.AsTensor().Reshape(new[] { 1, _embeddingDim }));
        hidden = engine.TensorMultiply(hidden, engine.TensorSigmoid(hidden));
        return engine.TensorAdd(
            engine.TensorMatMul(hidden, engine.TensorTranspose(Tensor<T>.FromMatrix(_outputWeights))),
            _outputBias.AsTensor().Reshape(new[] { 1, _embeddingDim }));
    }

    /// <summary>
    /// Gets parameters: first-layer weights (row-major) and bias, then second-layer weights and bias.
    /// </summary>
    public Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(new Vector<T>(_projectionWeights.AsSpan().ToArray()), _projectionBias),
            Vector<T>.Concatenate(new Vector<T>(_outputWeights.AsSpan().ToArray()), _outputBias));
    }

    /// <summary>
    /// Sets parameters in the <see cref="GetParameters"/> layout.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Camera embedding expects {ParameterCount} parameters but received {parameters.Length}.", nameof(parameters));
        var source = parameters.AsSpan();
        int idx = 0;
        source.Slice(idx, _projectionWeights.Rows * _projectionWeights.Columns).CopyTo(_projectionWeights.AsWritableSpan());
        idx += _projectionWeights.Rows * _projectionWeights.Columns;
        source.Slice(idx, _embeddingDim).CopyTo(_projectionBias.AsWritableSpan());
        idx += _embeddingDim;
        source.Slice(idx, _outputWeights.Rows * _outputWeights.Columns).CopyTo(_outputWeights.AsWritableSpan());
        idx += _outputWeights.Rows * _outputWeights.Columns;
        source.Slice(idx, _embeddingDim).CopyTo(_outputBias.AsWritableSpan());
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public long ParameterCount => (_embeddingDim * _inputDim) + _embeddingDim + (_embeddingDim * _embeddingDim) + _embeddingDim;
}
