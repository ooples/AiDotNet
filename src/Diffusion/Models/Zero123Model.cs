using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Models;

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
/// - Camera pose embedding via sinusoidal encoding
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
/// var inputImage = LoadImage("object.png");
/// var novelView = zero123.GenerateNovelView(
///     inputImage: inputImage,
///     polarAngle: 30.0,    // Rotate up 30 degrees
///     azimuthAngle: 45.0,  // Rotate right 45 degrees
///     radius: 1.0);        // Keep same distance
///
/// // Generate multiple views for 3D reconstruction
/// var views = zero123.Generate360Views(inputImage, numViews: 8);
/// </code>
/// </example>
public class Zero123Model<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// Standard latent channels.
    /// </summary>
    private const int Z123_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    private const int Z123_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Default image size.
    /// </summary>
    private const int DEFAULT_IMAGE_SIZE = 256;

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The image encoder for conditioning.
    /// </summary>
    private readonly ImageEncoder<T> _imageEncoder;

    /// <summary>
    /// The camera pose encoder.
    /// </summary>
    private readonly CameraPoseEncoder<T> _poseEncoder;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => Z123_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount =>
        _unet.ParameterCount + _vae.ParameterCount +
        _imageEncoder.ParameterCount + _poseEncoder.ParameterCount;

    /// <summary>
    /// Initializes a new instance of Zero123Model with default parameters.
    /// </summary>
    public Zero123Model()
        : this(
            options: null,
            scheduler: null,
            imageSize: DEFAULT_IMAGE_SIZE,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new instance of Zero123Model with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="imageSize">Image size for generation.</param>
    /// <param name="seed">Optional random seed.</param>
    public Zero123Model(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        int imageSize = DEFAULT_IMAGE_SIZE,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler())
    {
        // Create U-Net with image+pose conditioning
        _unet = unet ?? CreateDefaultUNet(seed);

        // Create VAE
        _vae = vae ?? CreateDefaultVAE(seed);

        // Create image encoder
        _imageEncoder = new ImageEncoder<T>(
            imageSize: 224,
            patchSize: 16,
            embedDim: 768,
            numLayers: 12,
            numHeads: 12,
            seed: seed);

        // Create camera pose encoder
        _poseEncoder = new CameraPoseEncoder<T>(
            embedDim: 768,
            seed: seed);
    }

    /// <summary>
    /// Creates the default options.
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
    /// Creates the default scheduler.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default U-Net.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(int? seed)
    {
        return new UNetNoisePredictor<T>(
            inputChannels: Z123_LATENT_CHANNELS + Z123_LATENT_CHANNELS, // Concat input latent
            outputChannels: Z123_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);
    }

    /// <summary>
    /// Creates the default VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: Z123_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

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
        double radius = 1.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Encode input image
        var imageEmbedding = _imageEncoder.Encode(inputImage);

        // Encode camera pose
        var poseEmbedding = _poseEncoder.Encode(polarAngle, azimuthAngle, radius);

        // Combine embeddings
        var conditioning = CombineEmbeddings(imageEmbedding, poseEmbedding);

        // Encode input image to latent
        var inputLatent = EncodeToLatent(inputImage);

        // Get dimensions
        var height = inputImage.Shape.Length > 2 ? inputImage.Shape[^2] : DEFAULT_IMAGE_SIZE;
        var width = inputImage.Shape[^1];
        var latentHeight = height / Z123_VAE_SCALE_FACTOR;
        var latentWidth = width / Z123_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, Z123_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
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
                // Unconditional with zero pose
                var zeroPose = _poseEncoder.Encode(0, 0, 1.0);
                var uncondConditioning = CombineEmbeddings(imageEmbedding, zeroPose);

                var condPred = _unet.PredictNoise(concatenatedLatent, timestep, conditioning);
                var uncondPred = _unet.PredictNoise(concatenatedLatent, timestep, uncondConditioning);
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
                radius: 1.0,
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
                radius: 1.0,
                numInferenceSteps,
                guidanceScale,
                viewSeed);

            views.Add(view);
        }

        return views;
    }

    /// <summary>
    /// Combines image and pose embeddings.
    /// </summary>
    private Tensor<T> CombineEmbeddings(Tensor<T> imageEmbed, Tensor<T> poseEmbed)
    {
        // Add embeddings (simplified combination)
        return AddTensors(imageEmbed, poseEmbed);
    }

    /// <summary>
    /// Concatenates two latent tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateLatents(Tensor<T> a, Tensor<T> b)
    {
        // Simplified: create tensor with doubled channels
        var newShape = new int[a.Shape.Length];
        Array.Copy(a.Shape, newShape, a.Shape.Length);
        newShape[1] *= 2; // Double channels

        var result = new Tensor<T>(newShape);
        var resultSpan = result.AsWritableSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        // Copy a then b
        for (int i = 0; i < aSpan.Length; i++)
        {
            resultSpan[i] = aSpan[i];
        }
        for (int i = 0; i < bSpan.Length && i + aSpan.Length < resultSpan.Length; i++)
        {
            resultSpan[aSpan.Length + i] = bSpan[i];
        }

        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var minLen = Math.Min(aSpan.Length, bSpan.Length);
        for (int i = 0; i < minLen; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddParams(allParams, _unet.GetParameters());
        AddParams(allParams, _vae.GetParameters());
        AddParams(allParams, _imageEncoder.GetParameters());
        AddParams(allParams, _poseEncoder.GetParameters());

        return new Vector<T>(allParams.ToArray());
    }

    private void AddParams(List<T> allParams, Vector<T> p)
    {
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        var unetCount = _unet.ParameterCount;
        var unetParams = new T[unetCount];
        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[offset++];
        }
        _unet.SetParameters(new Vector<T>(unetParams));

        var vaeCount = _vae.ParameterCount;
        var vaeParams = new T[vaeCount];
        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset++];
        }
        _vae.SetParameters(new Vector<T>(vaeParams));

        var encoderCount = _imageEncoder.ParameterCount;
        var encoderParams = new T[encoderCount];
        for (int i = 0; i < encoderCount; i++)
        {
            encoderParams[i] = parameters[offset++];
        }
        _imageEncoder.SetParameters(new Vector<T>(encoderParams));

        var poseCount = _poseEncoder.ParameterCount;
        var poseParams = new T[poseCount];
        for (int i = 0; i < poseCount; i++)
        {
            poseParams[i] = parameters[offset++];
        }
        _poseEncoder.SetParameters(new Vector<T>(poseParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new Zero123Model<T>(
            seed: RandomGenerator.Next());

        clone.SetParameters(GetParameters());
        return clone;
    }
}

/// <summary>
/// Encodes camera pose (polar, azimuth, radius) into embeddings.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CameraPoseEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _embedDim;
    private readonly DenseLayer<T> _projection;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount => _projection.ParameterCount;

    /// <summary>
    /// Initializes a new CameraPoseEncoder.
    /// </summary>
    public CameraPoseEncoder(int embedDim = 768, int? seed = null)
    {
        _embedDim = embedDim;
        // Input: sinusoidal encoding of polar, azimuth, radius (3 * 64 = 192)
        _projection = new DenseLayer<T>(192, embedDim, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Encodes camera pose into an embedding.
    /// </summary>
    /// <param name="polarAngle">Polar angle in degrees.</param>
    /// <param name="azimuthAngle">Azimuth angle in degrees.</param>
    /// <param name="radius">Relative radius.</param>
    /// <returns>Pose embedding tensor.</returns>
    public Tensor<T> Encode(double polarAngle, double azimuthAngle, double radius)
    {
        // Convert angles to radians
        var polar = polarAngle * Math.PI / 180.0;
        var azimuth = azimuthAngle * Math.PI / 180.0;

        // Create sinusoidal encoding for each value
        var encoding = new T[192]; // 64 dimensions per value * 3 values
        var idx = 0;

        // Encode polar angle
        for (int i = 0; i < 32; i++)
        {
            var freq = Math.Pow(10000, -2.0 * i / 64);
            encoding[idx++] = NumOps.FromDouble(Math.Sin(polar * freq));
            encoding[idx++] = NumOps.FromDouble(Math.Cos(polar * freq));
        }

        // Encode azimuth angle
        for (int i = 0; i < 32; i++)
        {
            var freq = Math.Pow(10000, -2.0 * i / 64);
            encoding[idx++] = NumOps.FromDouble(Math.Sin(azimuth * freq));
            encoding[idx++] = NumOps.FromDouble(Math.Cos(azimuth * freq));
        }

        // Encode radius (log scale)
        var logRadius = Math.Log(Math.Max(radius, 0.01));
        for (int i = 0; i < 32; i++)
        {
            var freq = Math.Pow(10000, -2.0 * i / 64);
            encoding[idx++] = NumOps.FromDouble(Math.Sin(logRadius * freq));
            encoding[idx++] = NumOps.FromDouble(Math.Cos(logRadius * freq));
        }

        var inputTensor = new Tensor<T>(new[] { 1, 192 }, new Vector<T>(encoding));
        return _projection.Forward(inputTensor);
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        return _projection.GetParameters();
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        _projection.SetParameters(parameters);
    }
}
