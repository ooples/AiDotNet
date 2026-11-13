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
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_multiViewUNet);
        RegisterParameterComponent(_imageVAE);
        RegisterParameterComponent(_cameraEmbedding);
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
    /// Camera embedding layer.
    /// </summary>
    private CameraEmbedding<T> _cameraEmbedding;

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
    /// Gets the camera embedding module.
    /// </summary>
    public CameraEmbedding<T> CameraEmbedding => _cameraEmbedding;

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
    /// <param name="imageConditioner">Optional image conditioning module.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="seed">Optional random seed.</param>
    public MVDreamModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MultiViewUNet<T>? multiViewUNet = null,
        StandardVAE<T>? imageVAE = null,
        IConditioningModule<T>? textConditioner = null,
        IConditioningModule<T>? imageConditioner = null,
        MVDreamConfig? config = null,
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
            defaultPointCount: 8192,
            architecture: architecture)
    {
        _config = config ?? MVDreamConfig.Default;
        _textConditioner = textConditioner;
        _imageConditioner = imageConditioner;

        InitializeLayers(multiViewUNet, imageVAE, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the model layers, using provided components or creating defaults.
    /// </summary>
    [MemberNotNull(nameof(_multiViewUNet), nameof(_imageVAE), nameof(_cameraEmbedding))]
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

        // MVDream adds the camera embedding to the timestep embedding, so it has the U-Net's time-embedding width.
        _cameraEmbedding = new CameraEmbedding<T>(
            embeddingDim: _multiViewUNet.BaseUNet.TimeEmbeddingDim,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates multiple consistent views from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the object.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numViews">Number of views to generate (4 or 8).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="elevation">Camera elevation angle in degrees.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Array of generated view images [numViews, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This generates multiple pictures of an object
    /// from different angles, all looking consistent:
    ///
    /// - numViews=4: Front, right, back, left (90-degree spacing)
    /// - numViews=8: Adds diagonal views (45-degree spacing)
    ///
    /// Higher elevation = looking more from above
    /// - 0 degrees: eye level
    /// - 30 degrees: looking down at 30-degree angle
    /// </para>
    /// </remarks>
    public virtual Tensor<T>[] GenerateMultiView(
        string prompt,
        string? negativePrompt = null,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        double elevation = 30.0,
        int? seed = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Multi-view generation requires a text conditioning module.");

        // Generate camera positions
        var cameraPositions = GenerateCameraPositions(numViews, elevation);

        // Encode text prompt
        var promptTokens = _textConditioner.Tokenize(prompt);
        var promptEmbedding = _textConditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (guidanceScale > 1.0)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _textConditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = _textConditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _textConditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Compute camera embeddings
        var cameraEmbeddings = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var (azimuth, elev, radius) = cameraPositions[v];
            cameraEmbeddings[v] = _cameraEmbedding.Embed(azimuth, elev, radius);
        }

        // Initialize latents for all views
        var latentH = MVDREAM_IMAGE_SIZE / 8;
        var latentW = MVDREAM_IMAGE_SIZE / 8;
        var latentShape = new[] { numViews, MVDREAM_LATENT_CHANNELS, latentH, latentW };

        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop with multi-view attention
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Text conditioning per view; the camera enters through the time embedding
            var combinedConditions = new Tensor<T>[numViews];
            for (int v = 0; v < numViews; v++)
            {
                combinedConditions[v] = promptEmbedding;
            }

            Tensor<T> noisePrediction;

            if (negativeEmbedding != null && guidanceScale > 1.0)
            {
                // CFG with multi-view
                var condPred = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions, cameraEmbeddings);

                var negConditions = new Tensor<T>[numViews];
                for (int v = 0; v < numViews; v++)
                {
                    negConditions[v] = negativeEmbedding;
                }
                var uncondPred = _multiViewUNet.PredictNoiseMultiView(latents, timestep, negConditions, cameraEmbeddings);

                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions, cameraEmbeddings);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
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

        // Reconstruct 3D from multi-view images
        var cameraPositions = GenerateCameraPositions(numViews, 30.0);
        return ReconstructFromMultiView(views, cameraPositions);
    }

    /// <summary>
    /// Generates novel views from a single input image.
    /// </summary>
    /// <param name="inputImage">Input image tensor.</param>
    /// <param name="numViews">Number of views to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Array of generated views.</returns>
    public virtual Tensor<T>[] GenerateNovelViewsFromImage(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        // Encode input image
        var imageLatent = _imageVAE.Encode(inputImage, sampleMode: false);

        // Generate camera positions (input is at front)
        var cameraPositions = GenerateCameraPositions(numViews, 30.0);

        // Get image conditioning if available
        Tensor<T>? imageCondition = null;
        if (_imageConditioner != null)
        {
            imageCondition = _imageConditioner.Encode(inputImage);
        }
        else
        {
            // Use VAE latent as conditioning
            imageCondition = FlattenLatent(imageLatent);
        }

        // Compute camera embeddings
        var cameraEmbeddings = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var (azimuth, elev, radius) = cameraPositions[v];
            cameraEmbeddings[v] = _cameraEmbedding.Embed(azimuth, elev, radius);
        }

        // Initialize latents
        var latentH = inputImage.Shape[2] / 8;
        var latentW = inputImage.Shape[3] / 8;
        var latentShape = new[] { numViews, MVDREAM_LATENT_CHANNELS, latentH, latentW };

        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set input view latent (first view is the input)
        SetView(latents, 0, _imageVAE.ScaleLatent(imageLatent));

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            var combinedConditions = new Tensor<T>[numViews];
            for (int v = 0; v < numViews; v++)
            {
                combinedConditions[v] = imageCondition;
            }

            var noisePrediction = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions, cameraEmbeddings);

            // Scheduler step (skip first view which is fixed)
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);

            // Restore input view
            SetView(latents, 0, _imageVAE.ScaleLatent(imageLatent));
        }

        // Decode views
        var views = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var viewLatent = ExtractView(latents, v);
            views[v] = _imageVAE.Decode(_imageVAE.UnscaleLatent(viewLatent));
        }

        return views;
    }

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
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Score distillation requires a text conditioning module.");

        var numViews = renderedViews.Shape[0];

        // Encode prompt
        var promptTokens = _textConditioner.Tokenize(prompt);
        var promptEmbedding = _textConditioner.EncodeText(promptTokens);
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

        // Generate camera embeddings for the views
        var cameraPositions = GenerateCameraPositions(numViews, 30.0);
        var cameraEmbeddings = new Tensor<T>[numViews];
        var condConditions = new Tensor<T>[numViews];
        var uncondConditions = new Tensor<T>[numViews];

        for (int v = 0; v < numViews; v++)
        {
            var (azimuth, elev, radius) = cameraPositions[v];
            cameraEmbeddings[v] = _cameraEmbedding.Embed(azimuth, elev, radius);
            condConditions[v] = promptEmbedding;
            uncondConditions[v] = uncondEmbedding;
        }

        // Predict noise with and without conditioning
        var condNoisePred = _multiViewUNet.PredictNoiseMultiView(noisyLatentsTensor, timestep, condConditions, cameraEmbeddings);
        var uncondNoisePred = _multiViewUNet.PredictNoiseMultiView(noisyLatentsTensor, timestep, uncondConditions, cameraEmbeddings);

        return ScoreDistillationGradient(condNoisePred, uncondNoisePred, noise, guidanceScale,
            Scheduler.GetAlphaCumulativeProduct(timestep));
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
    /// Flattens latent for conditioning.
    /// </summary>
    private Tensor<T> FlattenLatent(Tensor<T> latent)
    {
        var flat = latent.ToVector();
        return new Tensor<T>(new[] { 1, flat.Length }, flat);
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
    /// Transforms a point from camera space to world space.
    /// </summary>
    private (double x, double y, double z) TransformCameraToWorld(
        double camX, double camY, double camZ,
        double azimuth, double elevation, double radius)
    {
        // Camera position in world space
        double posX = radius * Math.Cos(elevation) * Math.Sin(azimuth);
        double posY = radius * Math.Sin(elevation);
        double posZ = radius * Math.Cos(elevation) * Math.Cos(azimuth);

        // Camera forward direction (pointing at origin)
        double fwdX = -posX, fwdY = -posY, fwdZ = -posZ;
        double fwdLen = Math.Sqrt(fwdX * fwdX + fwdY * fwdY + fwdZ * fwdZ);
        fwdX /= fwdLen; fwdY /= fwdLen; fwdZ /= fwdLen;

        // World up
        double upX = 0, upY = 1, upZ = 0;

        // Camera right
        double rightX = upY * fwdZ - upZ * fwdY;
        double rightY = upZ * fwdX - upX * fwdZ;
        double rightZ = upX * fwdY - upY * fwdX;
        double rightLen = Math.Sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
        if (rightLen > 1e-6)
        {
            rightX /= rightLen; rightY /= rightLen; rightZ /= rightLen;
        }
        else
        {
            // Camera looking straight up/down - use different up vector
            upX = 0; upY = 0; upZ = 1;
            rightX = upY * fwdZ - upZ * fwdY;
            rightY = upZ * fwdX - upX * fwdZ;
            rightZ = upX * fwdY - upY * fwdX;
            rightLen = Math.Sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
            rightX /= rightLen; rightY /= rightLen; rightZ /= rightLen;
        }

        // Camera up (recomputed to be orthogonal)
        double camUpX = fwdY * rightZ - fwdZ * rightY;
        double camUpY = fwdZ * rightX - fwdX * rightZ;
        double camUpZ = fwdX * rightY - fwdY * rightX;

        // Transform point from camera to world
        double worldX = posX + rightX * camX + camUpX * camY - fwdX * camZ;
        double worldY = posY + rightY * camX + camUpY * camY - fwdY * camZ;
        double worldZ = posZ + rightZ * camX + camUpZ * camY - fwdZ * camZ;

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
    /// Whether to use multi-view attention.
    /// </summary>
    public bool UseMultiViewAttention { get; set; } = true;

    /// <summary>
    /// Camera distance from object center.
    /// </summary>
    public double CameraDistance { get; set; } = 1.5;

    /// <summary>
    /// Default elevation angle in degrees.
    /// </summary>
    public double DefaultElevation { get; set; } = 30.0;

    /// <summary>
    /// Gets the default configuration.
    /// </summary>
    public static MVDreamConfig Default => new MVDreamConfig();
}

/// <summary>
/// Multi-view aware U-Net for MVDream.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class MultiViewUNet<T> : IParameterSource<T>, ICloneable<MultiViewUNet<T>>
{
    private readonly INumericOperations<T> _numOps;
    private UNetNoisePredictor<T> _baseUNet;
    private readonly int _numViews;
    private MultiViewAttention<T> _mvAttention;
    // Retained construction config so Clone() can create an identically-shaped instance around the
    // already-cloned resolved sub-modules (see MVDreamModel.Clone).
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _baseChannels;
    private readonly int _contextDim;

    /// <summary>
    /// Gets whether classifier-free guidance is supported.
    /// </summary>
    public bool SupportsCFG => true;

    /// <summary>
    /// Gets the base UNet for interface compatibility.
    /// </summary>
    public UNetNoisePredictor<T> BaseUNet => _baseUNet;

    /// <summary>
    /// Creates a new multi-view U-Net.
    /// </summary>
    public MultiViewUNet(
        int inputChannels,
        int outputChannels,
        int baseChannels,
        int numViews,
        int contextDim,
        int? seed = null)
        : this(
            inputChannels,
            outputChannels,
            baseChannels,
            numViews,
            contextDim,
            new UNetNoisePredictor<T>(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                baseChannels: baseChannels,
                channelMultipliers: new[] { 1, 2, 4, 4 },
                numResBlocks: 2,
                attentionResolutions: new[] { 4, 2, 1 },
                contextDim: contextDim,
                seed: seed),
            new MultiViewAttention<T>(
                channels: outputChannels, // Applied to the per-view predictions
                numViews: numViews,
                seed: seed))
    {
    }

    // Sole field-initialization path: the public constructor delegates here after
    // building its base U-Net and multi-view attention, and Clone() calls it directly
    // with the cloned sub-modules. Keeping a single constructor avoids duplicating the
    // (readonly) scalar-field assignments across both entry points.
    private MultiViewUNet(
        int inputChannels,
        int outputChannels,
        int baseChannels,
        int numViews,
        int contextDim,
        UNetNoisePredictor<T> baseUNet,
        MultiViewAttention<T> mvAttention)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numViews = numViews;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _baseChannels = baseChannels;
        _contextDim = contextDim;
        _baseUNet = baseUNet;
        _mvAttention = mvAttention;
    }

    /// <inheritdoc cref="Clone"/>
    public MultiViewUNet<T> DeepCopy() => Clone();

    /// <summary>
    /// Deep-copies this multi-view U-Net, preserving trained weights. The base U-Net is cloned via its
    /// own <c>Clone()</c> (which resolves and copies its lazy layers), and the multi-view attention block
    /// is rebuilt at the same shape and its parameters copied into the private clone constructor — so the
    /// result is structurally and observationally identical to this instance.
    /// </summary>
    // Declaring ICloneable is what lets CloneEngine find this Clone. Without it MVDreamModel's
    // configuration copy received the SOURCE's multi-view U-Net as its constructor argument, so the
    // clone and the original shared every base U-Net layer and Clone threw AliasedLayerGraph (#2087).
    public MultiViewUNet<T> Clone()
    {
        var clonedAttention = new MultiViewAttention<T>(channels: _outputChannels, numViews: _numViews);
        clonedAttention.SetParameters(_mvAttention.GetParameters());

        var clonedBaseUNet = (UNetNoisePredictor<T>)_baseUNet.Clone();

        // The attention block above is copied by value; the base U-Net must be too. Its own Clone
        // deliberately leaves an UNMATERIALIZED predictor lazy, which is right for the paper-scale
        // models that share that method and is not sufficient here: two lazy predictors each draw
        // their own weights the first time they are read, so the "clone" silently becomes a
        // different model. Measured on MultiViewUNet(baseChannels: 4), both sides report 96,140
        // parameters and 93,040 of them differ, starting at index 0.
        //
        // Confined to this type ON PURPOSE. Making UNetNoisePredictor.Clone resolve instead fixes
        // this and breaks four other tests, because they clone a predictor that must stay lazy:
        // StableDiffusion15Model_Clone_PreservesLazyParameterCount times out at 120 s and
        // DDPMModelTests.Clone_ReturnsNewInstance throws OutOfMemoryException out of Vector<T>'s
        // constructor. Nothing but MVDream builds a MultiViewUNet, so scoping the copy here leaves
        // those models exactly as they were.
        //
        // Chunks rather than the flat GetParameters/SetParameters pair: MVDream's real geometry is
        // baseChannels 320, and the chunk stream is the path built to move that much weight without
        // allocating a second copy of the whole surface.
        clonedBaseUNet.SetParameterChunks(_baseUNet.GetParameterChunks());

        return new MultiViewUNet<T>(
            _inputChannels,
            _outputChannels,
            _baseChannels,
            _numViews,
            _contextDim,
            clonedBaseUNet,
            clonedAttention);
    }

    /// <summary>
    /// Predicts noise for a single view.
    /// </summary>
    public Tensor<T> PredictNoise(Tensor<T> input, int timestep, Tensor<T>? conditioning = null)
    {
        return _baseUNet.PredictNoise(input, timestep, conditioning);
    }

    /// <summary>
    /// Predicts noise for multiple views with cross-view attention.
    /// </summary>
    public Tensor<T> PredictNoiseMultiView(Tensor<T> multiViewInput, int timestep, Tensor<T>[] viewConditions)
    {
        var numViews = multiViewInput.Shape[0];
        var viewPredictions = new Tensor<T>[numViews];

        // Process each view through base U-Net
        for (int v = 0; v < numViews; v++)
        {
            var viewInput = ExtractView(multiViewInput, v);
            var viewPred = _baseUNet.PredictNoise(viewInput, timestep, viewConditions[v]);
            viewPredictions[v] = viewPred;
        }

        // Apply multi-view attention for consistency
        var attended = _mvAttention.Apply(viewPredictions);

        // Combine back into multi-view tensor
        return CombineViews(attended, multiViewInput._shape);
    }

    /// <summary>
    /// Predicts noise for multiple views with cross-view attention, adding each view's camera embedding to the
    /// U-Net's time embedding (MVDream's camera conditioning).
    /// </summary>
    /// <param name="multiViewInput">The view latents [views, channels, height, width].</param>
    /// <param name="timestep">The timestep.</param>
    /// <param name="viewConditions">Per-view cross-attention conditioning.</param>
    /// <param name="cameraEmbeddings">Per-view camera embeddings of the U-Net's time-embedding width.</param>
    public Tensor<T> PredictNoiseMultiView(
        Tensor<T> multiViewInput, int timestep, Tensor<T>[] viewConditions, Tensor<T>[] cameraEmbeddings)
    {
        var numViews = multiViewInput.Shape[0];
        if (cameraEmbeddings.Length != numViews)
            throw new ArgumentException($"Expected {numViews} camera embeddings but received {cameraEmbeddings.Length}.",
                nameof(cameraEmbeddings));

        var viewPredictions = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var viewInput = ExtractView(multiViewInput, v);
            viewPredictions[v] = _baseUNet.PredictNoise(viewInput, timestep, viewConditions[v], cameraEmbeddings[v]);
        }

        // Apply multi-view attention for consistency
        var attended = _mvAttention.Apply(viewPredictions);
        return CombineViews(attended, multiViewInput._shape);
    }

    private static Tensor<T> ExtractView(Tensor<T> multiView, int viewIndex)
        => AiDotNetEngine.Current.TensorNarrow(multiView, 0, viewIndex, 1).Clone();

    private Tensor<T> CombineViews(Tensor<T>[] views, int[] shape)
    {
        return AiDotNetEngine.Current.TensorConcatenate<T>(views, axis: 0);
    }

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(_baseUNet.GetParameters(), _mvAttention.GetParameters());
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int unetCount = checked((int)_baseUNet.ParameterCount);
        int mvCount = checked((int)_mvAttention.ParameterCount);

        _baseUNet.SetParameters(parameters.GetSubVector(0, unetCount));
        _mvAttention.SetParameters(parameters.GetSubVector(unetCount, mvCount));
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public long ParameterCount => _baseUNet.ParameterCount + _mvAttention.ParameterCount;
}

/// <summary>
/// Multi-view attention module for cross-view consistency.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// MVDream (Shi et al., 2023) makes the views consistent by letting their tokens attend to one another. Here every
/// spatial position attends across the views with learned query/key/value/output projections of the channel
/// vectors it receives: <c>x + W_o softmax(Q K^T / sqrt(C)) V</c>, computed per position over the view axis.
/// </remarks>
public class MultiViewAttention<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Matrix<T> _queryWeights;
    private readonly Matrix<T> _keyWeights;
    private readonly Matrix<T> _valueWeights;
    private readonly Matrix<T> _outputWeights;
    private readonly int _channels;
    private readonly int _numViews;

    /// <summary>
    /// Creates multi-view attention.
    /// </summary>
    public MultiViewAttention(int channels, int numViews, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _channels = channels;
        _numViews = numViews;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var scale = Math.Sqrt(2.0 / channels);

        _queryWeights = new Matrix<T>(channels, channels);
        _keyWeights = new Matrix<T>(channels, channels);
        _valueWeights = new Matrix<T>(channels, channels);
        _outputWeights = new Matrix<T>(channels, channels);

        // Initialize weights
        InitializeMatrix(_queryWeights, rng, scale);
        InitializeMatrix(_keyWeights, rng, scale);
        InitializeMatrix(_valueWeights, rng, scale);
        InitializeMatrix(_outputWeights, rng, scale);
    }

    private static void InitializeMatrix(Matrix<T> matrix, Random rng, double scale)
    {
        // Row-major uniform draws in [-scale, scale), converted to T in one pass.
        var draws = new double[matrix.Rows * matrix.Columns];
        for (int i = 0; i < draws.Length; i++)
            draws[i] = (rng.NextDouble() * 2 - 1) * scale;
        new Tensor<double>(new[] { draws.Length }, new Vector<double>(draws)).Cast<T>().AsSpan()
            .CopyTo(matrix.AsWritableSpan());
    }

    /// <summary>
    /// Applies multi-view attention for cross-view consistency.
    /// </summary>
    /// <param name="views">One [1, channels, height, width] tensor per view.</param>
    public Tensor<T>[] Apply(Tensor<T>[] views)
    {
        if (views.Length == 0)
            return views;

        var engine = AiDotNetEngine.Current;
        var viewShape = views[0]._shape;
        int channels = viewShape[1];
        if (channels != _channels)
            throw new ArgumentException(
                $"Multi-view attention was built for {_channels} channels but the views have {channels}.", nameof(views));

        int numViews = views.Length;
        int positions = views[0].Length / channels;

        // tokens[p, v, c]: the channel vector of view v at position p.
        var stacked = engine.TensorConcatenate(views, 0).Reshape(new[] { numViews, channels, positions });
        var tokens = engine.TensorPermute(stacked, new[] { 2, 0, 1 });
        var flat = tokens.Reshape(new[] { positions * numViews, channels });

        var query = Project(engine, flat, _queryWeights).Reshape(new[] { positions, numViews, channels });
        var key = Project(engine, flat, _keyWeights).Reshape(new[] { positions, numViews, channels });
        var value = Project(engine, flat, _valueWeights).Reshape(new[] { positions, numViews, channels });

        var scores = engine.TensorMultiplyScalar(
            engine.BatchMatMul(query, engine.TensorPermute(key, new[] { 0, 2, 1 })),
            _numOps.FromDouble(1.0 / Math.Sqrt(channels)));
        var attended = engine.BatchMatMul(engine.Softmax(scores, 2), value);
        var projected = Project(engine, attended.Reshape(new[] { positions * numViews, channels }), _outputWeights);

        var updated = engine.TensorAdd(flat, projected).Reshape(new[] { positions, numViews, channels });
        var perView = engine.TensorPermute(updated, new[] { 1, 2, 0 });

        var result = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
            result[v] = engine.TensorNarrow(perView, 0, v, 1).Reshape(views[v]._shape).Clone();
        return result;
    }

    /// <summary>
    /// x W^T for row vectors x.
    /// </summary>
    private static Tensor<T> Project(IEngine engine, Tensor<T> rows, Matrix<T> weights)
        => engine.TensorMatMul(rows, engine.TensorTranspose(Tensor<T>.FromMatrix(weights)));

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var total = _channels * _channels * 4;
        var result = new Vector<T>(total);

        int idx = 0;
        CopyMatrixToVector(_queryWeights, result, ref idx);
        CopyMatrixToVector(_keyWeights, result, ref idx);
        CopyMatrixToVector(_valueWeights, result, ref idx);
        CopyMatrixToVector(_outputWeights, result, ref idx);

        return result;
    }

    private static void CopyMatrixToVector(Matrix<T> matrix, Vector<T> vector, ref int idx)
    {
        int count = matrix.Rows * matrix.Columns;
        matrix.AsSpan().CopyTo(vector.AsWritableSpan().Slice(idx, count));
        idx += count;
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        CopyVectorToMatrix(parameters, _queryWeights, ref idx);
        CopyVectorToMatrix(parameters, _keyWeights, ref idx);
        CopyVectorToMatrix(parameters, _valueWeights, ref idx);
        CopyVectorToMatrix(parameters, _outputWeights, ref idx);
    }

    private static void CopyVectorToMatrix(Vector<T> vector, Matrix<T> matrix, ref int idx)
    {
        int count = matrix.Rows * matrix.Columns;
        vector.AsSpan().Slice(idx, count).CopyTo(matrix.AsWritableSpan());
        idx += count;
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public long ParameterCount => _channels * _channels * 4;
}

/// <summary>
/// Camera embedding for view conditioning.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// As in MVDream (Shi et al., 2023), the camera is described by its flattened 4x4 camera-to-world extrinsic
/// matrix (16 values) and embedded by a two-layer MLP (linear, SiLU, linear) whose output is added to the U-Net's
/// timestep embedding.
/// </remarks>
public class CameraEmbedding<T> : IParameterSource<T>
{
    /// <summary>
    /// Size of the camera description: the flattened 4x4 extrinsic matrix.
    /// </summary>
    public const int CameraDim = 16;

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
    public static double[] CameraToWorld(double azimuth, double elevation, double radius)
    {
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
    /// Embeds a camera position: <c>W_2 SiLU(W_1 vec(E) + b_1) + b_2</c> for its extrinsic matrix E.
    /// </summary>
    /// <returns>The embedding [1, embeddingDim].</returns>
    public Tensor<T> Embed(double azimuth, double elevation, double radius)
    {
        var engine = AiDotNetEngine.Current;
        var extrinsic = new Tensor<double>(new[] { 1, CameraDim }, new Vector<double>(CameraToWorld(azimuth, elevation, radius)))
            .Cast<T>();
        var hidden = engine.TensorAdd(
            engine.TensorMatMul(extrinsic, engine.TensorTranspose(Tensor<T>.FromMatrix(_projectionWeights))),
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
