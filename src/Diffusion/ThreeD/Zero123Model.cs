using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Control;
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
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Zero-1-to-3 model for novel view synthesis from a single image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Zero-1-to-3 (Zero123) generates new viewpoints of an object from just a single
/// input image. It uses camera pose conditioning to control the viewpoint change,
/// enabling 3D-aware image generation without explicit 3D reconstruction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Zero123 is like having a magical camera that can show you
/// what an object looks like from different angles, even though you only gave it
/// one photograph.
///
/// What it does:
/// - Takes a single image of an object
/// - Generates images of that same object from different viewpoints
/// - Works with any object: cars, furniture, animals, etc.
///
/// Input parameters:
/// - Image: The original photo of the object
/// - Camera rotation: How much to rotate the view (polar/azimuth angles)
/// - Scale change: How close/far to zoom
///
/// Use cases:
/// - E-commerce: Show products from multiple angles
/// - 3D reconstruction: Generate training data for 3D models
/// - AR/VR: Create object previews from any angle
/// - Game development: Generate sprite variations
/// </para>
/// <para>
/// Technical details:
/// - Fine-tuned from Stable Diffusion
/// - Uses CLIP image encoder for conditioning
/// - Camera conditioning as in the official code: the CLIP image embedding concatenated with
///   [delta polar (rad), sin delta azimuth, cos delta azimuth, delta radius] through cc_projection
///   (Linear 772 -> 768, identity-initialised), plus the input image's VAE latent concatenated to the
///   denoiser input; classifier-free guidance zeroes both
/// - Supports arbitrary viewpoint changes
/// - Can be used iteratively for 360° reconstruction
///
/// Reference: Liu et al., "Zero-1-to-3: Zero-shot One Image to 3D Object", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Zero123 model
/// var zero123 = new Zero123Model&lt;float&gt;();
///
/// // Generate novel view
/// var inputImage = ImageHelper&lt;float&gt;.LoadImage("object.png");
/// var novelView = zero123.GenerateNovelView(
///     inputImage: inputImage,
///     polarAngle: 30.0,    // Polar change in degrees
///     azimuthAngle: 45.0,  // Azimuth change in degrees
///     radius: 0.0);        // Same distance (radius change 0)
///
/// // Generate multiple views for 3D reconstruction
/// var views = zero123.Generate360Views(inputImage, numViews: 8);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Zero-1-to-3: Zero-shot One Image to 3D Object", "https://arxiv.org/abs/2303.11328", Year = 2023, Authors = "Liu et al.")]
public partial class Zero123Model<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
        RegisterParameterComponent(_imageEncoder);
        RegisterParameterComponent(_cameraProjection);
    }

    #region Constants

    /// <summary>
    /// Number of latent channels in the VAE representation.
    /// </summary>
    /// <remarks>
    /// Zero123 uses the same 4-channel latent space as Stable Diffusion,
    /// encoding color, texture, edge, and structural information.
    /// </remarks>
    private const int Z123_LATENT_CHANNELS = 4;

    /// <summary>
    /// Spatial downsampling factor of the VAE.
    /// </summary>
    /// <remarks>
    /// The VAE compresses images by 8x spatially, so a 256x256 image
    /// becomes a 32x32 latent representation.
    /// </remarks>
    private const int Z123_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Default image size for Zero123 generation.
    /// </summary>
    /// <remarks>
    /// Zero123 was trained on 256x256 images. This is the recommended
    /// resolution for best results.
    /// </remarks>
    private const int DEFAULT_IMAGE_SIZE = 256;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The image encoder for conditioning.
    /// </summary>
    private ImageEncoder<T> _imageEncoder;

    /// <summary>
    /// The camera pose encoder.
    /// </summary>
    private ClipCameraProjection<T> _cameraProjection;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => Z123_LATENT_CHANNELS;

    /// <inheritdoc />

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Zero123Model with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="imageSize">Image size for generation.</param>
    /// <param name="seed">Optional random seed.</param>
    public Zero123Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        int imageSize = DEFAULT_IMAGE_SIZE,
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
            architecture)
    {
        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net, VAE, image encoder, and camera pose encoder.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae), nameof(_imageEncoder), nameof(_cameraProjection))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // U-Net with doubled input channels for concatenated input latent
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: Z123_LATENT_CHANNELS + Z123_LATENT_CHANNELS,
            outputChannels: Z123_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            architecture: Architecture,
            seed: seed);

        // Standard SD 1.5 VAE
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: Z123_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        // CLIP image encoder for conditioning
        _imageEncoder = new ImageEncoder<T>(
            imageSize: 224,
            patchSize: 16,
            embedDim: 768,
            numLayers: 12,
            numHeads: 12,
            seed: seed);

        // Camera pose encoder for viewpoint control
        // cc_projection: Linear(768 + 4 -> 768), identity-initialised as the official code does.
        _cameraProjection = new ClipCameraProjection<T>(embedDim: 768, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates a novel view of an object.
    /// </summary>
    /// <param name="inputImage">The input image of the object.</param>
    /// <param name="polarAngle">Polar angle change in degrees (vertical rotation).</param>
    /// <param name="azimuthAngle">Azimuth angle change in degrees (horizontal rotation).</param>
    /// <param name="radius">Relative radius change (1.0 = same distance).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The generated novel view image.</returns>
    public virtual Tensor<T> GenerateNovelView(
        Tensor<T> inputImage,
        double polarAngle,
        double azimuthAngle,
        double radius = 0.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        // The image embedding [1, 1, D] and the camera-conditioned projection of it (cc_projection).
        var imageEmbedding = EncodeImageEmbedding(inputImage);
        var conditioning = _cameraProjection.Project(imageEmbedding, new[] { new GuidanceViewOffset(polarAngle, azimuthAngle, radius) });

        // Encode input image to latent
        var inputLatent = EncodeToLatent(inputImage);

        // Get dimensions
        var height = inputImage.Shape.Length > 2 ? inputImage.Shape[^2] : DEFAULT_IMAGE_SIZE;
        var width = inputImage.Shape[^1];
        var latentHeight = height / Z123_VAE_SCALE_FACTOR;
        var latentWidth = width / Z123_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, Z123_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Concatenate input latent with noise latent
            var concatenatedLatent = ConcatenateLatents(latents, inputLatent);

            Tensor<T> noisePrediction;

            if (effectiveGuidanceScale > 1.0)
            {
                // The official unconditional branch zeroes both the projected embedding and the image latent.
                var uncondConditioning = new Tensor<T>(conditioning._shape);
                var uncondLatent = ConcatenateLatents(latents, new Tensor<T>(inputLatent._shape));

                var condPred = _unet.PredictNoise(concatenatedLatent, timestep, conditioning);
                var uncondPred = _unet.PredictNoise(uncondLatent, timestep, uncondConditioning);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(concatenatedLatent, timestep, conditioning);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Generates multiple views around an object (360° views).
    /// </summary>
    /// <param name="inputImage">The input image.</param>
    /// <param name="numViews">Number of views to generate.</param>
    /// <param name="polarAngle">Fixed polar angle for all views.</param>
    /// <param name="numInferenceSteps">Denoising steps per view.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>List of generated views.</returns>
    public virtual List<Tensor<T>> Generate360Views(
        Tensor<T> inputImage,
        int numViews = 8,
        double polarAngle = 0.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var views = new List<Tensor<T>>();
        var azimuthStep = 360.0 / numViews;

        for (int i = 0; i < numViews; i++)
        {
            var azimuth = i * azimuthStep;
            int? viewSeed = seed.HasValue ? seed.Value + i : (int?)null;

            var view = GenerateNovelView(
                inputImage,
                polarAngle,
                azimuth,
                radius: 0.0,
                numInferenceSteps,
                guidanceScale,
                viewSeed);

            views.Add(view);
        }

        return views;
    }

    /// <summary>
    /// Generates views at multiple elevation angles.
    /// </summary>
    /// <param name="inputImage">The input image.</param>
    /// <param name="azimuthAngles">List of azimuth angles.</param>
    /// <param name="polarAngles">List of polar angles.</param>
    /// <param name="numInferenceSteps">Denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>List of generated views.</returns>
    public virtual List<Tensor<T>> GenerateMultipleViews(
        Tensor<T> inputImage,
        double[] azimuthAngles,
        double[] polarAngles,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var views = new List<Tensor<T>>();

        for (int i = 0; i < azimuthAngles.Length; i++)
        {
            var polar = i < polarAngles.Length ? polarAngles[i] : 0.0;
            int? viewSeed = seed.HasValue ? seed.Value + i : (int?)null;

            var view = GenerateNovelView(
                inputImage,
                polar,
                azimuthAngles[i],
                radius: 0.0,
                numInferenceSteps,
                guidanceScale,
                viewSeed);

            views.Add(view);
        }

        return views;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// The image embedding Zero-1-to-3 conditions on, [1, 1, D]: the image encoder's tokens mean-pooled into one
    /// vector (the role of CLIP's pooled, projected <c>image_embeds</c>).
    /// </summary>
    internal Tensor<T> EncodeImageEmbedding(Tensor<T> image)
    {
        var tokens = _imageEncoder.Encode(image);
        int dim = tokens._shape[tokens._shape.Length - 1];
        var flat = Engine.Reshape(tokens, new[] { tokens.Length / dim, dim });
        var pooled = Engine.ReduceMean(flat, new[] { 0 }, true);
        return Engine.Reshape(pooled, new[] { 1, 1, dim });
    }

    /// <summary>The camera-conditioned projection (<c>cc_projection</c>).</summary>
    internal ClipCameraProjection<T> CameraProjection => _cameraProjection;

    /// <summary>
    /// Loads the official <c>cc_projection</c> weights (<c>weight</c> [768, 772] and <c>bias</c> [768], PyTorch
    /// <c>nn.Linear</c> layout) from a state-dict reader.
    /// </summary>
    /// <param name="read">Reads a named tensor of the given shape as row-major values.</param>
    /// <param name="prefix">The key prefix: <c>cc_projection.</c> in the original checkpoint,
    /// <c>clip_camera_projection.proj.</c> in the diffusers conversion.</param>
    public void LoadCameraProjectionWeights(Func<string, int[], double[]> read, string prefix = "cc_projection.")
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        if (prefix is null) throw new ArgumentNullException(nameof(prefix));
        int d = _cameraProjection.EmbeddingDimension;
        var weight = read(prefix + "weight", new[] { d, d + ClipCameraProjection<T>.CameraFeatureCount });
        var bias = read(prefix + "bias", new[] { d });
        _cameraProjection.LoadWeights(weight, bias);
    }

    /// <summary>
    /// Concatenates two latent tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateLatents(Tensor<T> a, Tensor<T> b)
    {
        // Concatenate along channel dimension (axis 1) using hardware-accelerated engine
        return Engine.TensorConcatenate<T>(new[] { a, b }, axis: 1);
    }

    #endregion

    #region IParameterizable Implementation



    #endregion

    #region ICloneable Implementation

    #endregion
}

/// <summary>
/// Zero-1-to-3's camera-conditioned projection (<c>cc_projection</c> in the official code,
/// <c>CLIPCameraProjection</c> in the diffusers conversion): the image embedding concatenated with the relative camera
/// <c>[delta polar (radians), sin(delta azimuth), cos(delta azimuth), delta radius]</c> and mapped back to the embedding
/// width by one linear layer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The official initialisation (<c>ldm/models/diffusion/ddpm.py</c>) sets the embedding block of the weight to the
/// identity (<c>nn.init.eye_(weight[:768, :768])</c>) and the bias to zero; the four camera columns keep PyTorch's
/// default <c>nn.Linear</c> initialisation, U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
/// </para>
/// <para><b>For Beginners:</b> this small layer mixes "what the object looks like" with "how far the camera moved", so
/// the denoiser knows which view to draw.</para>
/// </remarks>
public class ClipCameraProjection<T> : IParameterSource<T>
{
    /// <summary>The number of camera features appended to the embedding (4).</summary>
    public const int CameraFeatureCount = 4;

    private readonly int _embedDim;
    private Tensor<T> _weight;
    private Tensor<T> _bias;

    /// <summary>Creates the projection with the official initialisation.</summary>
    /// <param name="embedDim">The image-embedding width (768 for CLIP ViT-L/14).</param>
    /// <param name="seed">Seed of the camera columns' default initialisation.</param>
    public ClipCameraProjection(int embedDim = 768, int? seed = null)
    {
        if (embedDim < 1) throw new ArgumentOutOfRangeException(nameof(embedDim), "embedDim must be positive.");
        _embedDim = embedDim;
        int inputs = embedDim + CameraFeatureCount;
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        double bound = 1.0 / Math.Sqrt(inputs);
        var weight = new double[embedDim * inputs];
        for (int row = 0; row < embedDim; row++)
        {
            weight[(row * inputs) + row] = 1.0;
            for (int k = 0; k < CameraFeatureCount; k++)
                weight[(row * inputs) + embedDim + k] = ((2.0 * random.NextDouble()) - 1.0) * bound;
        }

        _weight = new Tensor<double>(new[] { embedDim, inputs }, new Vector<double>(weight)).Cast<T>();
        _bias = new Tensor<T>(new[] { embedDim });
    }

    /// <summary>The embedding width.</summary>
    public int EmbeddingDimension => _embedDim;

    /// <inheritdoc />
    public long ParameterCount => ((long)_embedDim * (_embedDim + CameraFeatureCount)) + _embedDim;

    /// <summary>
    /// The camera features of a relative view, <c>[deg2rad(delta polar), sin(deg2rad(delta azimuth)),
    /// cos(deg2rad(delta azimuth)), delta radius]</c>.
    /// </summary>
    public static double[] CameraFeatures(GuidanceViewOffset view)
    {
        double polar = view.Elevation * Math.PI / 180.0;
        double azimuth = view.Azimuth * Math.PI / 180.0;
        return new[] { polar, Math.Sin(azimuth), Math.Cos(azimuth), view.Radius };
    }

    /// <summary>
    /// Projects the image embedding for each view: <c>W [e; c] + b</c>.
    /// </summary>
    /// <param name="imageEmbedding">The image embedding, [1, 1, D] (repeated for every view) or [B, 1, D].</param>
    /// <param name="views">The relative cameras, one per output.</param>
    /// <returns>The conditioning, [B, 1, D].</returns>
    public Tensor<T> Project(Tensor<T> imageEmbedding, IReadOnlyList<GuidanceViewOffset> views)
    {
        if (imageEmbedding is null) throw new ArgumentNullException(nameof(imageEmbedding));
        if (views is null) throw new ArgumentNullException(nameof(views));
        int batch = views.Count;
        if (batch < 1) throw new ArgumentException("At least one view is needed.", nameof(views));
        if (imageEmbedding.Length % _embedDim != 0)
            throw new ArgumentException($"The image embedding must have width {_embedDim}.", nameof(imageEmbedding));
        int given = imageEmbedding.Length / _embedDim;
        if (given != 1 && given != batch)
            throw new ArgumentException("Give one image embedding, or one per view.", nameof(imageEmbedding));

        var engine = AiDotNetEngine.Current;
        var embedding = engine.Reshape(imageEmbedding, new[] { given, _embedDim });
        if (given != batch)
            embedding = engine.TensorBroadcastTo(embedding, new[] { batch, _embedDim });
        var cameras = new double[batch * CameraFeatureCount];
        for (int b = 0; b < batch; b++)
            Array.Copy(CameraFeatures(views[b]), 0, cameras, b * CameraFeatureCount, CameraFeatureCount);
        var camera = new Tensor<double>(new[] { batch, CameraFeatureCount }, new Vector<double>(cameras)).Cast<T>();
        var input = engine.TensorConcatenate(new[] { embedding, camera }, 1);
        var projected = engine.TensorMatMulTransposed(input, _weight);
        var bias = engine.TensorBroadcastTo(engine.Reshape(_bias, new[] { 1, _embedDim }), new[] { batch, _embedDim });
        return engine.Reshape(engine.TensorAdd(projected, bias), new[] { batch, 1, _embedDim });
    }

    /// <summary>
    /// Loads official weights in PyTorch <c>nn.Linear</c> layout: <paramref name="weight"/> [D, D + 4] row-major and
    /// <paramref name="bias"/> [D].
    /// </summary>
    public void LoadWeights(double[] weight, double[] bias)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (bias is null) throw new ArgumentNullException(nameof(bias));
        int inputs = _embedDim + CameraFeatureCount;
        if (weight.Length != _embedDim * inputs)
            throw new ArgumentException($"The weight must be [{_embedDim}, {inputs}].", nameof(weight));
        if (bias.Length != _embedDim)
            throw new ArgumentException($"The bias must be [{_embedDim}].", nameof(bias));
        _weight = new Tensor<double>(new[] { _embedDim, inputs }, new Vector<double>((double[])weight.Clone())).Cast<T>();
        _bias = new Tensor<double>(new[] { _embedDim }, new Vector<double>((double[])bias.Clone())).Cast<T>();
    }

    /// <inheritdoc />
    /// <remarks>The weight (row-major, PyTorch layout) followed by the bias.</remarks>
    public Vector<T> GetParameters()
    {
        var weight = _weight.ToArray();
        var bias = _bias.ToArray();
        var all = new T[weight.Length + bias.Length];
        Array.Copy(weight, all, weight.Length);
        Array.Copy(bias, 0, all, weight.Length, bias.Length);
        return new Vector<T>(all);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        var values = parameters.ToArray();
        int weightCount = _embedDim * (_embedDim + CameraFeatureCount);
        var weight = new T[weightCount];
        var bias = new T[_embedDim];
        Array.Copy(values, weight, weightCount);
        Array.Copy(values, weightCount, bias, 0, _embedDim);
        _weight = new Tensor<T>(new[] { _embedDim, _embedDim + CameraFeatureCount }, new Vector<T>(weight));
        _bias = new Tensor<T>(new[] { _embedDim }, new Vector<T>(bias));
    }
}
