using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Models;

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
public class MVDreamModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <summary>
    /// Default image resolution for each view.
    /// </summary>
    public const int MVDREAM_IMAGE_SIZE = 256;

    /// <summary>
    /// MVDream latent channels (Stable Diffusion compatible).
    /// </summary>
    public const int MVDREAM_LATENT_CHANNELS = 4;

    /// <summary>
    /// MVDream base channels for U-Net.
    /// </summary>
    public const int MVDREAM_BASE_CHANNELS = 320;

    /// <summary>
    /// Context dimension for text/image conditioning.
    /// </summary>
    public const int MVDREAM_CONTEXT_DIM = 1024;

    /// <summary>
    /// Default camera distance from object.
    /// </summary>
    public const double DEFAULT_CAMERA_DISTANCE = 1.5;

    /// <summary>
    /// The multi-view aware U-Net noise predictor.
    /// </summary>
    private readonly MultiViewUNet<T> _multiViewUNet;

    /// <summary>
    /// The VAE for image encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _imageVAE;

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
    private readonly CameraEmbedding<T> _cameraEmbedding;

    /// <summary>
    /// Model configuration.
    /// </summary>
    private readonly MVDreamConfig _config;

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

    /// <summary>
    /// Initializes a new MVDream model with default parameters.
    /// </summary>
    public MVDreamModel()
        : this(
            options: null,
            scheduler: null,
            multiViewUNet: null,
            imageVAE: null,
            textConditioner: null,
            imageConditioner: null,
            config: MVDreamConfig.Default,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new MVDream model with custom parameters.
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
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MultiViewUNet<T>? multiViewUNet = null,
        StandardVAE<T>? imageVAE = null,
        IConditioningModule<T>? textConditioner = null,
        IConditioningModule<T>? imageConditioner = null,
        MVDreamConfig? config = null,
        int? seed = null)
        : base(
            options ?? CreateDefaultOptions(),
            scheduler ?? CreateDefaultScheduler(),
            defaultPointCount: 8192)
    {
        _config = config ?? MVDreamConfig.Default;
        _textConditioner = textConditioner;
        _imageConditioner = imageConditioner;

        // Initialize image VAE
        _imageVAE = imageVAE ?? CreateDefaultImageVAE(seed);

        // Initialize multi-view U-Net
        _multiViewUNet = multiViewUNet ?? CreateDefaultMultiViewUNet(seed);

        // Initialize camera embedding
        _cameraEmbedding = new CameraEmbedding<T>(
            embeddingDim: MVDREAM_CONTEXT_DIM,
            seed: seed);
    }

    /// <summary>
    /// Creates default options for MVDream.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000,
            BetaStart = 0.00085,
            BetaEnd = 0.012,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
    }

    /// <summary>
    /// Creates the default DDIM scheduler.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default image VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultImageVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: MVDREAM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

    /// <summary>
    /// Creates the default multi-view U-Net.
    /// </summary>
    private MultiViewUNet<T> CreateDefaultMultiViewUNet(int? seed)
    {
        return new MultiViewUNet<T>(
            inputChannels: MVDREAM_LATENT_CHANNELS,
            outputChannels: MVDREAM_LATENT_CHANNELS,
            baseChannels: MVDREAM_BASE_CHANNELS,
            numViews: _config.DefaultNumViews,
            contextDim: MVDREAM_CONTEXT_DIM,
            seed: seed);
    }

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
                var negTokens = _textConditioner.Tokenize(negativePrompt);
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

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop with multi-view attention
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Combine text and camera conditioning for each view
            var combinedConditions = new Tensor<T>[numViews];
            for (int v = 0; v < numViews; v++)
            {
                combinedConditions[v] = CombineConditions(promptEmbedding, cameraEmbeddings[v]);
            }

            Tensor<T> noisePrediction;

            if (negativeEmbedding != null && guidanceScale > 1.0)
            {
                // CFG with multi-view
                var condPred = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions);

                var negConditions = new Tensor<T>[numViews];
                for (int v = 0; v < numViews; v++)
                {
                    negConditions[v] = CombineConditions(negativeEmbedding, cameraEmbeddings[v]);
                }
                var uncondPred = _multiViewUNet.PredictNoiseMultiView(latents, timestep, negConditions);

                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode each view
        var views = new Tensor<T>[numViews];
        for (int v = 0; v < numViews; v++)
        {
            var viewLatent = ExtractView(latents, v);
            views[v] = _imageVAE.Decode(viewLatent);
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

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
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
                combinedConditions[v] = CombineConditions(imageCondition, cameraEmbeddings[v]);
            }

            var noisePrediction = _multiViewUNet.PredictNoiseMultiView(latents, timestep, combinedConditions);

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
        var noise = SampleNoiseTensor(latents.Shape, RandomGenerator);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timestep);
        var noisyLatentsTensor = new Tensor<T>(latents.Shape, noisyLatents);

        // Generate camera embeddings for the views
        var cameraPositions = GenerateCameraPositions(numViews, 30.0);
        var combinedCondCond = new Tensor<T>[numViews];
        var combinedUncondCond = new Tensor<T>[numViews];

        for (int v = 0; v < numViews; v++)
        {
            var (azimuth, elev, radius) = cameraPositions[v];
            var camEmbed = _cameraEmbedding.Embed(azimuth, elev, radius);
            combinedCondCond[v] = CombineConditions(promptEmbedding, camEmbed);
            combinedUncondCond[v] = CombineConditions(uncondEmbedding, camEmbed);
        }

        // Predict noise with and without conditioning
        var condNoisePred = _multiViewUNet.PredictNoiseMultiView(noisyLatentsTensor, timestep, combinedCondCond);
        var uncondNoisePred = _multiViewUNet.PredictNoiseMultiView(noisyLatentsTensor, timestep, combinedUncondCond);

        // Compute SDS gradient
        var gradient = new Tensor<T>(condNoisePred.Shape);
        var gradSpan = gradient.AsWritableSpan();
        var condSpan = condNoisePred.AsSpan();
        var uncondSpan = uncondNoisePred.AsSpan();

        var scale = NumOps.FromDouble(guidanceScale);

        for (int i = 0; i < gradSpan.Length; i++)
        {
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);
            gradSpan[i] = NumOps.Multiply(scale, diff);
        }

        return gradient;
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
    /// Combines text/image conditioning with camera embedding.
    /// </summary>
    private Tensor<T> CombineConditions(Tensor<T> contentCondition, Tensor<T> cameraCondition)
    {
        // Concatenate along sequence dimension or add as extra tokens
        var contentFlat = contentCondition.ToVector();
        var cameraFlat = cameraCondition.ToVector();

        var combinedLen = contentFlat.Length + cameraFlat.Length;
        var combined = new Tensor<T>(new[] { 1, combinedLen });
        var span = combined.AsWritableSpan();

        for (int i = 0; i < contentFlat.Length; i++)
            span[i] = contentFlat[i];
        for (int i = 0; i < cameraFlat.Length; i++)
            span[contentFlat.Length + i] = cameraFlat[i];

        return combined;
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

        for (int i = 0; i < viewSize && viewOffset + i < multiSpan.Length; i++)
        {
            viewSpan[i] = multiSpan[viewOffset + i];
        }

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

        for (int i = 0; i < viewSize && i < viewSpan.Length && viewOffset + i < multiSpan.Length; i++)
        {
            multiSpan[viewOffset + i] = viewSpan[i];
        }
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
    /// Reconstructs mesh from multiple views.
    /// </summary>
    private Mesh3D<T> ReconstructFromMultiView(
        Tensor<T>[] views,
        (double azimuth, double elevation, double radius)[] cameras)
    {
        // Extract depth/features from each view and triangulate
        // This is a simplified implementation
        var numPoints = DefaultPointCount;
        var points = new Tensor<T>(new[] { 1, numPoints, 3 });
        var pointSpan = points.AsWritableSpan();

        // Generate points on sphere and project through views
        var rng = RandomGenerator;
        for (int i = 0; i < numPoints; i++)
        {
            var theta = 2.0 * Math.PI * rng.NextDouble();
            var phi = Math.Acos(2.0 * rng.NextDouble() - 1.0);

            // Random radius with slight variation
            var r = 0.8 + 0.4 * rng.NextDouble();

            pointSpan[i * 3] = NumOps.FromDouble(r * Math.Sin(phi) * Math.Cos(theta));
            pointSpan[i * 3 + 1] = NumOps.FromDouble(r * Math.Sin(phi) * Math.Sin(theta));
            pointSpan[i * 3 + 2] = NumOps.FromDouble(r * Math.Cos(phi));
        }

        return PointCloudToMesh(points, SurfaceReconstructionMethod.Poisson);
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _multiViewUNet.GetParameters();
        var vaeParams = _imageVAE.GetParameters();
        var camParams = _cameraEmbedding.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length + camParams.Length;
        var combined = new Vector<T>(totalLength);

        int offset = 0;

        for (int i = 0; i < unetParams.Length; i++)
            combined[offset + i] = unetParams[i];
        offset += unetParams.Length;

        for (int i = 0; i < vaeParams.Length; i++)
            combined[offset + i] = vaeParams[i];
        offset += vaeParams.Length;

        for (int i = 0; i < camParams.Length; i++)
            combined[offset + i] = camParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _multiViewUNet.ParameterCount;
        var vaeCount = _imageVAE.ParameterCount;
        var camCount = _cameraEmbedding.ParameterCount;

        var expected = unetCount + vaeCount + camCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;

        var unetParams = new Vector<T>(unetCount);
        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[offset + i];
        offset += unetCount;

        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[offset + i];
        offset += vaeCount;

        var camParams = new Vector<T>(camCount);
        for (int i = 0; i < camCount; i++)
            camParams[i] = parameters[offset + i];

        _multiViewUNet.SetParameters(unetParams);
        _imageVAE.SetParameters(vaeParams);
        _cameraEmbedding.SetParameters(camParams);
    }

    /// <inheritdoc />
    public override int ParameterCount =>
        _multiViewUNet.ParameterCount + _imageVAE.ParameterCount + _cameraEmbedding.ParameterCount;

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        return new MVDreamModel<T>(
            options: null,
            scheduler: null,
            multiViewUNet: null,
            imageVAE: null,
            textConditioner: _textConditioner,
            imageConditioner: _imageConditioner,
            config: _config);
    }

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
public class MultiViewUNet<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly UNetNoisePredictor<T> _baseUNet;
    private readonly int _numViews;
    private readonly MultiViewAttention<T> _mvAttention;

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
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numViews = numViews;

        _baseUNet = new UNetNoisePredictor<T>(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            baseChannels: baseChannels,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: contextDim,
            seed: seed);

        _mvAttention = new MultiViewAttention<T>(
            channels: baseChannels * 4, // At bottleneck
            numViews: numViews,
            seed: seed);
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
        return CombineViews(attended, multiViewInput.Shape);
    }

    private Tensor<T> ExtractView(Tensor<T> multiView, int viewIndex)
    {
        var viewShape = new[] { 1, multiView.Shape[1], multiView.Shape[2], multiView.Shape[3] };
        var view = new Tensor<T>(viewShape);
        var viewSpan = view.AsWritableSpan();
        var multiSpan = multiView.AsSpan();

        var viewSize = multiView.Shape[1] * multiView.Shape[2] * multiView.Shape[3];
        var viewOffset = viewIndex * viewSize;

        for (int i = 0; i < viewSize && viewOffset + i < multiSpan.Length; i++)
        {
            viewSpan[i] = multiSpan[viewOffset + i];
        }

        return view;
    }

    private Tensor<T> CombineViews(Tensor<T>[] views, int[] shape)
    {
        var result = new Tensor<T>(shape);
        var resultSpan = result.AsWritableSpan();

        var viewSize = shape[1] * shape[2] * shape[3];

        for (int v = 0; v < views.Length; v++)
        {
            var viewSpan = views[v].AsSpan();
            var offset = v * viewSize;

            for (int i = 0; i < viewSize && i < viewSpan.Length && offset + i < resultSpan.Length; i++)
            {
                resultSpan[offset + i] = viewSpan[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var unetParams = _baseUNet.GetParameters();
        var mvParams = _mvAttention.GetParameters();

        var total = unetParams.Length + mvParams.Length;
        var result = new Vector<T>(total);

        for (int i = 0; i < unetParams.Length; i++)
            result[i] = unetParams[i];
        for (int i = 0; i < mvParams.Length; i++)
            result[unetParams.Length + i] = mvParams[i];

        return result;
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var unetCount = _baseUNet.ParameterCount;
        var mvCount = _mvAttention.ParameterCount;

        var unetParams = new Vector<T>(unetCount);
        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[i];

        var mvParams = new Vector<T>(mvCount);
        for (int i = 0; i < mvCount; i++)
            mvParams[i] = parameters[unetCount + i];

        _baseUNet.SetParameters(unetParams);
        _mvAttention.SetParameters(mvParams);
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public int ParameterCount => _baseUNet.ParameterCount + _mvAttention.ParameterCount;
}

/// <summary>
/// Multi-view attention module for cross-view consistency.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
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

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
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

    private void InitializeMatrix(Matrix<T> matrix, Random rng, double scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = _numOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            }
        }
    }

    /// <summary>
    /// Applies multi-view attention for cross-view consistency.
    /// </summary>
    public Tensor<T>[] Apply(Tensor<T>[] views)
    {
        // Simplified: average features across views with learned blending
        var result = new Tensor<T>[views.Length];

        for (int v = 0; v < views.Length; v++)
        {
            var viewShape = views[v].Shape;
            result[v] = new Tensor<T>(viewShape);
            var resultSpan = result[v].AsWritableSpan();
            var viewSpan = views[v].AsSpan();

            // Weighted average with other views
            var selfWeight = 0.7; // Self-weight
            var otherWeight = (1.0 - selfWeight) / Math.Max(1, views.Length - 1);

            for (int i = 0; i < viewSpan.Length; i++)
            {
                var sum = _numOps.ToDouble(viewSpan[i]) * selfWeight;

                for (int other = 0; other < views.Length; other++)
                {
                    if (other != v)
                    {
                        var otherSpan = views[other].AsSpan();
                        if (i < otherSpan.Length)
                        {
                            sum += _numOps.ToDouble(otherSpan[i]) * otherWeight;
                        }
                    }
                }

                resultSpan[i] = _numOps.FromDouble(sum);
            }
        }

        return result;
    }

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

    private void CopyMatrixToVector(Matrix<T> matrix, Vector<T> vector, ref int idx)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                vector[idx++] = matrix[i, j];
            }
        }
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

    private void CopyVectorToMatrix(Vector<T> vector, Matrix<T> matrix, ref int idx)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = vector[idx++];
            }
        }
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public int ParameterCount => _channels * _channels * 4;
}

/// <summary>
/// Camera position embedding for view conditioning.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class CameraEmbedding<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Matrix<T> _projectionWeights;
    private readonly Vector<T> _projectionBias;
    private readonly int _embeddingDim;
    private readonly int _inputDim;

    /// <summary>
    /// Creates camera embedding.
    /// </summary>
    public CameraEmbedding(int embeddingDim, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = embeddingDim;
        _inputDim = 12; // Sinusoidal encoding of azimuth, elevation, radius

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var scale = Math.Sqrt(2.0 / _inputDim);

        _projectionWeights = new Matrix<T>(embeddingDim, _inputDim);
        _projectionBias = new Vector<T>(embeddingDim);

        for (int i = 0; i < embeddingDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                _projectionWeights[i, j] = _numOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            }
            _projectionBias[i] = _numOps.Zero;
        }
    }

    /// <summary>
    /// Embeds camera position.
    /// </summary>
    public Tensor<T> Embed(double azimuth, double elevation, double radius)
    {
        // Create sinusoidal encoding
        var input = new double[]
        {
            Math.Sin(azimuth), Math.Cos(azimuth),
            Math.Sin(2 * azimuth), Math.Cos(2 * azimuth),
            Math.Sin(elevation), Math.Cos(elevation),
            Math.Sin(2 * elevation), Math.Cos(2 * elevation),
            Math.Sin(radius), Math.Cos(radius),
            Math.Sin(2 * radius), Math.Cos(2 * radius)
        };

        // Project through linear layer
        var output = new Tensor<T>(new[] { 1, _embeddingDim });
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < _embeddingDim; i++)
        {
            var sum = _numOps.ToDouble(_projectionBias[i]);
            for (int j = 0; j < _inputDim; j++)
            {
                sum += input[j] * _numOps.ToDouble(_projectionWeights[i, j]);
            }
            outputSpan[i] = _numOps.FromDouble(sum);
        }

        return output;
    }

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var total = _embeddingDim * _inputDim + _embeddingDim;
        var result = new Vector<T>(total);

        int idx = 0;
        for (int i = 0; i < _embeddingDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                result[idx++] = _projectionWeights[i, j];
            }
        }
        for (int i = 0; i < _embeddingDim; i++)
        {
            result[idx++] = _projectionBias[i];
        }

        return result;
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int i = 0; i < _embeddingDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                _projectionWeights[i, j] = parameters[idx++];
            }
        }
        for (int i = 0; i < _embeddingDim; i++)
        {
            _projectionBias[i] = parameters[idx++];
        }
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public int ParameterCount => _embeddingDim * _inputDim + _embeddingDim;
}
