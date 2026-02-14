using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Stable Diffusion XL (SDXL) model for high-resolution image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SDXL is Stability AI's flagship text-to-image model, designed for
/// high-quality 1024x1024 image generation with improved prompt understanding
/// and visual fidelity compared to earlier Stable Diffusion versions.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDXL is like Stable Diffusion 2.0 but significantly upgraded:
///
/// Key improvements over SD 1.5/2.0:
/// - 4x larger U-Net (2.6B vs 865M parameters)
/// - Dual text encoders (better prompt understanding)
/// - Native 1024x1024 resolution (vs 512x512)
/// - Optional refiner model for enhanced details
///
/// How SDXL works:
/// 1. Your prompt goes through TWO text encoders (CLIP + OpenCLIP)
/// 2. These embeddings guide a much larger U-Net during denoising
/// 3. The base model generates at 1024x1024
/// 4. (Optional) A refiner model enhances fine details
///
/// Example prompt flow:
/// "A majestic dragon" -> [CLIP] + [OpenCLIP] -> Combined embedding
/// -> Large U-Net denoises -> 1024x1024 image
/// -> (Optional) Refiner -> Enhanced details
///
/// Use SDXL when you need:
/// - High resolution output
/// - Better text rendering in images
/// - More detailed and coherent images
/// - Following complex prompts accurately
/// </para>
/// <para>
/// Technical specifications:
/// - Base model: 2.6B parameter U-Net
/// - Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-bigG/14
/// - Native resolution: 1024x1024
/// - Latent space: 4 channels, 8x spatial downsampling
/// - Guidance scale: 5.0-9.0 recommended (7.5 default)
/// - Scheduler: DDPM/DPM++/Euler with 20-50 steps
///
/// Architecture details:
/// - Micro-conditioning: Size and crop coordinates for multi-aspect training
/// - Dual text encoding: Concatenated CLIP + OpenCLIP embeddings
/// - Channel multipliers: [1, 2, 4, 4] (vs [1, 2, 4, 8] in SD 2.x)
/// - Cross-attention dimension: 2048 (vs 1024 in SD 1.x)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an SDXL model
/// var sdxl = new SDXLModel&lt;float&gt;();
///
/// // Generate a high-resolution image
/// var image = sdxl.GenerateFromText(
///     prompt: "A majestic dragon perched on a mountain peak at sunset, highly detailed",
///     negativePrompt: "blurry, low quality, distorted",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5,
///     seed: 42);
///
/// // Generate with micro-conditioning for aspect ratio
/// var wideImage = sdxl.GenerateWithMicroCondition(
///     prompt: "Panoramic landscape with mountains and lake",
///     width: 1536,
///     height: 640,
///     originalWidth: 1536,
///     originalHeight: 640,
///     cropTop: 0,
///     cropLeft: 0);
///
/// // Use the refiner for enhanced details
/// if (sdxl.SupportsRefiner)
/// {
///     var refined = sdxl.RefineImage(image, "enhance details");
/// }
/// </code>
/// </example>
public class SDXLModel<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// Default width for SDXL generation.
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default height for SDXL generation.
    /// </summary>
    public const int DefaultHeight = 1024;

    /// <summary>
    /// Standard SDXL latent channels.
    /// </summary>
    private const int SDXL_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard SDXL VAE scale factor.
    /// </summary>
    private const int SDXL_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The primary conditioning module (CLIP ViT-L).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner1;

    /// <summary>
    /// The secondary conditioning module (OpenCLIP ViT-bigG).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner2;

    /// <summary>
    /// Optional refiner model.
    /// </summary>
    private readonly SDXLRefiner<T>? _refiner;

    /// <summary>
    /// Whether to use dual text encoders.
    /// </summary>
    private readonly bool _useDualEncoder;

    /// <summary>
    /// Cross-attention dimension for SDXL (2048).
    /// </summary>
    private readonly int _crossAttentionDim;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner1;

    /// <inheritdoc />
    public override int LatentChannels => SDXL_LATENT_CHANNELS;

    /// <summary>
    /// Gets the secondary text encoder if available.
    /// </summary>
    public IConditioningModule<T>? SecondaryConditioner => _conditioner2;

    /// <summary>
    /// Gets whether this model uses dual text encoders.
    /// </summary>
    public bool UsesDualEncoder => _useDualEncoder;

    /// <summary>
    /// Gets whether this model has a refiner available.
    /// </summary>
    public bool SupportsRefiner => _refiner != null;

    /// <summary>
    /// Gets the refiner model if available.
    /// </summary>
    public SDXLRefiner<T>? Refiner => _refiner;

    /// <summary>
    /// Gets the cross-attention dimension (2048 for SDXL).
    /// </summary>
    public int CrossAttentionDim => _crossAttentionDim;

    /// <summary>
    /// Initializes a new instance of SDXLModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner1">Optional primary text encoder (CLIP).</param>
    /// <param name="conditioner2">Optional secondary text encoder (OpenCLIP).</param>
    /// <param name="refiner">Optional refiner model.</param>
    /// <param name="useDualEncoder">Whether to use dual text encoders.</param>
    /// <param name="crossAttentionDim">Cross-attention dimension (2048 for SDXL).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SDXLModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner1 = null,
        IConditioningModule<T>? conditioner2 = null,
        SDXLRefiner<T>? refiner = null,
        bool useDualEncoder = true,
        int crossAttentionDim = 2048,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler(), architecture)
    {
        _crossAttentionDim = crossAttentionDim;
        _useDualEncoder = useDualEncoder;
        _refiner = refiner;

        // Initialize U-Net with SDXL-specific parameters
        _unet = unet ?? CreateDefaultUNet(seed);

        // Initialize VAE
        _vae = vae ?? CreateDefaultVAE(seed);

        // Initialize conditioning modules
        _conditioner1 = conditioner1;
        _conditioner2 = useDualEncoder ? conditioner2 : null;
    }

    /// <summary>
    /// Creates default options for SDXL.
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
    /// Creates the default DDIM scheduler for SDXL.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default U-Net for SDXL (2.6B parameters).
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(int? seed)
    {
        // SDXL uses larger U-Net with [1, 2, 4, 4] channel multipliers
        return new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: SDXL_LATENT_CHANNELS,
            outputChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 }, // Different from SD 1.x
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: _crossAttentionDim, // 2048 for SDXL
            seed: seed);
    }

    /// <summary>
    /// Creates the default VAE for SDXL.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

    /// <summary>
    /// Generates an image with micro-conditioning for multi-aspect ratio support.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt to guide away from.</param>
    /// <param name="width">Output image width.</param>
    /// <param name="height">Output image height.</param>
    /// <param name="originalWidth">Original target width for conditioning.</param>
    /// <param name="originalHeight">Original target height for conditioning.</param>
    /// <param name="cropTop">Top crop coordinate for conditioning.</param>
    /// <param name="cropLeft">Left crop coordinate for conditioning.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Micro-conditioning helps SDXL generate better images
    /// at various aspect ratios by telling the model about the target size and
    /// any cropping applied during training.
    /// </para>
    /// <para>
    /// When generating at non-square resolutions:
    /// - Set originalWidth/originalHeight to your target size
    /// - Set cropTop/cropLeft to 0 for centered generation
    /// - The model adjusts its generation accordingly
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateWithMicroCondition(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int? originalWidth = null,
        int? originalHeight = null,
        int cropTop = 0,
        int cropLeft = 0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Text-to-image generation requires a conditioning module.");

        // Default to actual size if not specified
        originalWidth ??= width;
        originalHeight ??= height;

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && _unet.SupportsCFG;

        // Encode text with dual encoder if available
        var promptEmbedding = EncodeTextDual(prompt);
        Tensor<T>? negativeEmbedding = null;

        if (useCFG)
        {
            negativeEmbedding = !string.IsNullOrEmpty(negativePrompt)
                ? EncodeTextDual(negativePrompt ?? string.Empty)
                : GetUnconditionalEmbeddingDual();
        }

        // Create micro-conditioning vector
        var microCond = CreateMicroCondition(
            originalWidth.Value, originalHeight.Value,
            cropTop, cropLeft,
            width, height);

        // Add micro-conditioning to embeddings
        promptEmbedding = ApplyMicroCondition(promptEmbedding, microCond);
        if (negativeEmbedding != null)
        {
            negativeEmbedding = ApplyMicroCondition(negativeEmbedding, microCond);
        }

        // Calculate latent dimensions
        var latentHeight = height / SDXL_VAE_SCALE_FACTOR;
        var latentWidth = width / SDXL_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, SDXL_LATENT_CHANNELS, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = _unet.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = _unet.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latents, timestep, promptEmbedding);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to image
        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Refines an image using the SDXL refiner model.
    /// </summary>
    /// <param name="image">The base image to refine.</param>
    /// <param name="prompt">The text prompt (should match base generation).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of refiner steps (typically 20-30).</param>
    /// <param name="denoiseStrength">How much to denoise (0.2-0.4 typical for refining).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Refined image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The refiner is a specialized model that takes
    /// an already-generated image and enhances fine details:
    ///
    /// Without refiner:
    /// - Base SDXL generates good overall structure
    /// - Some fine details may be slightly soft
    ///
    /// With refiner:
    /// - Details like skin texture, fabric, hair are enhanced
    /// - Overall coherence is preserved
    /// - Image looks more "finished"
    ///
    /// Best practices:
    /// - Use denoiseStrength 0.2-0.4 (higher = more change)
    /// - Use 20-30 refiner steps
    /// - Keep the same prompt as base generation
    /// </para>
    /// </remarks>
    public virtual Tensor<T> RefineImage(
        Tensor<T> image,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 25,
        double denoiseStrength = 0.3,
        int? seed = null)
    {
        if (_refiner == null)
            throw new InvalidOperationException("Refiner model not available. Initialize SDXLModel with a refiner.");

        return _refiner.Refine(
            image,
            prompt,
            negativePrompt,
            numInferenceSteps,
            denoiseStrength,
            seed);
    }

    /// <summary>
    /// Encodes text using dual text encoders.
    /// </summary>
    private Tensor<T> EncodeTextDual(string prompt)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Primary conditioner not initialized.");

        var tokens1 = _conditioner1.Tokenize(prompt);
        var embedding1 = _conditioner1.EncodeText(tokens1);

        if (_useDualEncoder && _conditioner2 != null)
        {
            var tokens2 = _conditioner2.Tokenize(prompt);
            var embedding2 = _conditioner2.EncodeText(tokens2);

            // Concatenate embeddings from both encoders
            return ConcatenateEmbeddings(embedding1, embedding2);
        }

        return embedding1;
    }

    /// <summary>
    /// Gets unconditional embedding for CFG with dual encoders.
    /// </summary>
    private Tensor<T> GetUnconditionalEmbeddingDual()
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Primary conditioner not initialized.");

        var uncond1 = _conditioner1.GetUnconditionalEmbedding(1);

        if (_useDualEncoder && _conditioner2 != null)
        {
            var uncond2 = _conditioner2.GetUnconditionalEmbedding(1);
            return ConcatenateEmbeddings(uncond1, uncond2);
        }

        return uncond1;
    }

    /// <summary>
    /// Concatenates embeddings from two text encoders.
    /// </summary>
    private Tensor<T> ConcatenateEmbeddings(Tensor<T> embedding1, Tensor<T> embedding2)
    {
        var shape1 = embedding1.Shape;
        var shape2 = embedding2.Shape;

        // Concatenate along the embedding dimension
        var batch = shape1[0];
        var seqLen = shape1[1];
        var dim1 = shape1[2];
        var dim2 = shape2[2];
        var totalDim = dim1 + dim2;

        var result = new Tensor<T>(new[] { batch, seqLen, totalDim });
        var resultSpan = result.AsWritableSpan();
        var span1 = embedding1.AsSpan();
        var span2 = embedding2.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Copy from first embedding
                for (int d = 0; d < dim1; d++)
                {
                    var srcIdx = b * seqLen * dim1 + s * dim1 + d;
                    var dstIdx = b * seqLen * totalDim + s * totalDim + d;
                    resultSpan[dstIdx] = span1[srcIdx];
                }

                // Copy from second embedding
                for (int d = 0; d < dim2; d++)
                {
                    var srcIdx = b * seqLen * dim2 + s * dim2 + d;
                    var dstIdx = b * seqLen * totalDim + s * totalDim + dim1 + d;
                    resultSpan[dstIdx] = span2[srcIdx];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Creates micro-conditioning vector for aspect ratio handling.
    /// </summary>
    private Tensor<T> CreateMicroCondition(
        int originalWidth, int originalHeight,
        int cropTop, int cropLeft,
        int targetWidth, int targetHeight)
    {
        // SDXL micro-conditioning: [original_size, crop_coords, target_size]
        var microCond = new Tensor<T>(new[] { 1, 6 });
        var span = microCond.AsWritableSpan();

        span[0] = NumOps.FromDouble(originalWidth);
        span[1] = NumOps.FromDouble(originalHeight);
        span[2] = NumOps.FromDouble(cropTop);
        span[3] = NumOps.FromDouble(cropLeft);
        span[4] = NumOps.FromDouble(targetWidth);
        span[5] = NumOps.FromDouble(targetHeight);

        return microCond;
    }

    /// <summary>
    /// Concatenates micro-conditioning to text embedding.
    /// </summary>
    private Tensor<T> ApplyMicroCondition(Tensor<T> embedding, Tensor<T> microCond)
    {
        // SDXL uses micro-conditioning (original_size, crop_coords, target_size) to improve
        // generation quality at different resolutions. The 6 values are projected through
        // a learned embedding and added to influence the diffusion process.
        var shape = embedding.Shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var embedDim = shape[2];

        var result = new Tensor<T>(new[] { batch, seqLen, embedDim });
        var resultSpan = result.AsWritableSpan();
        var embSpan = embedding.AsSpan();
        var microSpan = microCond.AsSpan();

        // Project micro-conditioning values (6 values) to embedding dimension
        // Each micro-cond value is scaled and used as a modulation factor
        int microLen = microCond.Shape[1]; // Should be 6

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < embedDim; d++)
                {
                    int embIdx = b * seqLen * embedDim + s * embedDim + d;
                    T embVal = embSpan[embIdx];

                    // Apply micro-conditioning as sinusoidal modulation
                    // Similar to how timestep embedding works in diffusion models
                    int microIdx = d % microLen;
                    T microVal = microSpan[b * microLen + microIdx];

                    // Scale micro-conditioning (normalized to prevent explosion)
                    double microDouble = NumOps.ToDouble(microVal);
                    double normalizedMicro = microDouble / 1024.0; // Normalize by typical image size
                    double modulation = Math.Sin(d * normalizedMicro * 0.01);

                    // Apply additive modulation
                    resultSpan[embIdx] = NumOps.Add(embVal, NumOps.FromDouble(modulation * 0.1));
                }
            }
        }

        return result;
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Combine parameters from U-Net and VAE
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.");

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

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
        // Clone U-Net with trained weights
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: SDXL_LATENT_CHANNELS,
            outputChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: _crossAttentionDim);
        clonedUnet.SetParameters(_unet.GetParameters());

        // Clone VAE with trained weights
        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2);
        clonedVae.SetParameters(_vae.GetParameters());

        return new SDXLModel<T>(
            options: null,
            scheduler: null,
            unet: clonedUnet,
            vae: clonedVae,
            conditioner1: _conditioner1,
            conditioner2: _conditioner2,
            refiner: _refiner,
            useDualEncoder: _useDualEncoder,
            crossAttentionDim: _crossAttentionDim);
    }

    #endregion
}

/// <summary>
/// SDXL Refiner model for enhancing generated images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SDXL refiner is a specialized model trained to enhance the fine details
/// of images generated by the SDXL base model. It operates in the latent space
/// and is designed to work with partially denoised latents.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of the refiner as a "finishing touch" step:
///
/// Without refiner:
/// Base image (good structure, okay details)
///
/// With refiner:
/// Refined image (same structure, enhanced details like texture, sharpness)
///
/// The refiner uses lower noise levels (typically 0.2-0.4 denoising strength)
/// to preserve the overall composition while enhancing fine details.
/// </para>
/// </remarks>
public class SDXLRefiner<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The refiner U-Net.
    /// </summary>
    private readonly UNetNoisePredictor<T> _refinerUNet;

    /// <summary>
    /// The VAE (shared with base model).
    /// </summary>
    private readonly IVAEModel<T> _vae;

    /// <summary>
    /// The scheduler.
    /// </summary>
    private readonly INoiseScheduler<T> _scheduler;

    /// <summary>
    /// The conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new SDXL refiner.
    /// </summary>
    /// <param name="refinerUNet">The refiner U-Net.</param>
    /// <param name="vae">The VAE (shared with base model).</param>
    /// <param name="scheduler">The scheduler.</param>
    /// <param name="conditioner">Optional conditioning module.</param>
    /// <param name="seed">Optional random seed.</param>
    public SDXLRefiner(
        UNetNoisePredictor<T> refinerUNet,
        IVAEModel<T> vae,
        INoiseScheduler<T> scheduler,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
    {
        _refinerUNet = refinerUNet;
        _vae = vae;
        _scheduler = scheduler;
        _conditioner = conditioner;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Refines an image by enhancing details.
    /// </summary>
    /// <param name="image">The input image to refine.</param>
    /// <param name="prompt">Text prompt describing the image.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of refiner steps.</param>
    /// <param name="denoiseStrength">Denoising strength (0.0-1.0).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Refined image tensor.</returns>
    public Tensor<T> Refine(
        Tensor<T> image,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 25,
        double denoiseStrength = 0.3,
        int? seed = null)
    {
        // Encode image to latent
        var latent = _vae.Encode(image, sampleMode: false);
        latent = _vae.ScaleLatent(latent);

        // Encode text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_conditioner != null)
        {
            var tokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(tokens);

            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = _conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Set up scheduler for refining (partial denoising)
        _scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - denoiseStrength));

        // Add noise at the starting point
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : _random;
        var noise = SampleNoise(latent.Shape, rng);

        var startTimestep = _scheduler.Timesteps.Skip(startStep).FirstOrDefault();
        latent = AddNoiseAtTimestep(latent, noise, startTimestep);

        // Denoising loop (only the remaining steps)
        foreach (var timestep in _scheduler.Timesteps.Skip(startStep))
        {
            Tensor<T> noisePrediction;

            if (promptEmbedding != null && negativeEmbedding != null)
            {
                var condPred = _refinerUNet.PredictNoise(latent, timestep, promptEmbedding);
                var uncondPred = _refinerUNet.PredictNoise(latent, timestep, negativeEmbedding);

                // Apply guidance (typically lower for refiner, around 5.0)
                noisePrediction = ApplyGuidance(uncondPred, condPred, 5.0);
            }
            else
            {
                noisePrediction = _refinerUNet.PredictNoise(latent, timestep, null);
            }

            // Scheduler step
            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = _scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent.Shape, latentVector);
        }

        // Decode back to image
        var unscaled = _vae.UnscaleLatent(latent);
        return _vae.Decode(unscaled);
    }

    /// <summary>
    /// Samples noise tensor.
    /// </summary>
    private Tensor<T> SampleNoise(int[] shape, Random rng)
    {
        var noise = new Tensor<T>(shape);
        var span = noise.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    /// <summary>
    /// Adds noise at a specific timestep.
    /// </summary>
    private Tensor<T> AddNoiseAtTimestep(Tensor<T> latent, Tensor<T> noise, int timestep)
    {
        // Simplified noise addition based on timestep
        // In practice, this uses the scheduler's alpha values
        var alpha = 1.0 - (timestep / 1000.0);
        var sigma = Math.Sqrt(1.0 - alpha * alpha);

        var result = new Tensor<T>(latent.Shape);
        var resultSpan = result.AsWritableSpan();
        var latentSpan = latent.AsSpan();
        var noiseSpan = noise.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var latentVal = NumOps.ToDouble(latentSpan[i]);
            var noiseVal = NumOps.ToDouble(noiseSpan[i]);
            resultSpan[i] = NumOps.FromDouble(alpha * latentVal + sigma * noiseVal);
        }

        return result;
    }

    /// <summary>
    /// Applies classifier-free guidance.
    /// </summary>
    private Tensor<T> ApplyGuidance(Tensor<T> uncondPred, Tensor<T> condPred, double guidanceScale)
    {
        var result = new Tensor<T>(condPred.Shape);
        var resultSpan = result.AsWritableSpan();
        var uncondSpan = uncondPred.AsSpan();
        var condSpan = condPred.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var uncondVal = NumOps.ToDouble(uncondSpan[i]);
            var condVal = NumOps.ToDouble(condSpan[i]);
            var guided = uncondVal + guidanceScale * (condVal - uncondVal);
            resultSpan[i] = NumOps.FromDouble(guided);
        }

        return result;
    }
}
