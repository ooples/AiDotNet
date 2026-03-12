using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Stable Diffusion 1.5 model for text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Diffusion 1.5 (SD 1.5) is a latent diffusion model developed by Stability AI and
/// Runway ML. It is the most widely used open-source text-to-image model and the foundation
/// for an enormous ecosystem of fine-tunes, LoRAs, ControlNets, and community models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stable Diffusion 1.5 is the "classic" text-to-image AI model.
///
/// How SD 1.5 works:
/// 1. Your text prompt is encoded by a CLIP ViT-L/14 text encoder into embeddings
/// 2. These embeddings guide a U-Net (865M parameters) that denoises in latent space
/// 3. A VAE decodes the denoised latent into a 512x512 image
///
/// Key characteristics:
/// - Single text encoder: CLIP ViT-L/14 (768-dim embeddings)
/// - U-Net: 865M parameters, channel multipliers [1, 2, 4, 4]
/// - VAE: 4-channel latent space, 8x spatial downsampling
/// - Native resolution: 512x512 pixels
/// - Latent scale factor: 0.18215
/// - Guidance scale: 7.5 (default)
///
/// When to use SD 1.5:
/// - Huge community model ecosystem (thousands of fine-tunes available)
/// - Lower resource requirements than SDXL (runs on 4GB+ VRAM)
/// - Fast generation (20-50 steps)
/// - Excellent for 512x512 generation
///
/// Limitations:
/// - Lower resolution than SDXL (512x512 vs 1024x1024)
/// - Single text encoder (less prompt understanding than dual-encoder models)
/// - Occasional artifacts at hands, text, and complex scenes
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: U-Net with cross-attention and time embedding
/// - Text encoder: CLIP ViT-L/14 (63M parameters, 768-dim, 77 max tokens)
/// - U-Net: 865M parameters, base channels 320, [1, 2, 4, 4] multipliers
/// - Cross-attention dimension: 768 (matches CLIP output)
/// - Attention resolutions: at 4x, 2x, and 1x downsampling levels
/// - VAE: KL-regularized autoencoder, 4 latent channels, scale factor 0.18215
/// - Noise schedule: Scaled linear beta schedule, 1000 training timesteps
/// - Beta range: [0.00085, 0.012]
/// - Prediction type: Epsilon (noise prediction)
///
/// Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var sd15 = new StableDiffusion15Model&lt;float&gt;();
///
/// // Create with custom components for full customization
/// var customSd15 = new StableDiffusion15Model&lt;float&gt;(
///     unet: myCustomUNet,
///     vae: myCustomVAE,
///     conditioner: myClipEncoder);
///
/// // Generate a 512x512 image from text
/// var image = sd15.GenerateFromText(
///     prompt: "A photograph of a cat sitting on a windowsill at sunset",
///     negativePrompt: "blurry, low quality, distorted",
///     width: 512,
///     height: 512,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5,
///     seed: 42);
///
/// // Image-to-image transformation
/// var transformed = sd15.ImageToImage(
///     inputImage: existingImage,
///     prompt: "Oil painting style, vibrant colors",
///     strength: 0.7,
///     numInferenceSteps: 30);
///
/// // Inpainting
/// var inpainted = sd15.Inpaint(
///     inputImage: existingImage,
///     mask: maskTensor,
///     prompt: "A fluffy golden retriever",
///     numInferenceSteps: 30);
/// </code>
/// </example>
public class StableDiffusion15Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SD 1.5 generation.
    /// </summary>
    /// <remarks>
    /// SD 1.5 was trained on 512x512 images. Generating at this resolution
    /// produces the best results. Other resolutions (e.g., 768x512) are possible
    /// but may introduce artifacts.
    /// </remarks>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for SD 1.5 generation.
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels in SD 1.5's VAE.
    /// </summary>
    /// <remarks>
    /// The VAE compresses 3-channel RGB images into a 4-channel latent representation.
    /// These 4 channels encode color, texture, edge, and structural information
    /// in a learned compressed form.
    /// </remarks>
    private const int SD15_LATENT_CHANNELS = 4;

    /// <summary>
    /// Spatial downsampling factor of the VAE (512 / 8 = 64).
    /// </summary>
    private const int SD15_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-L/14 output (768).
    /// </summary>
    /// <remarks>
    /// This must match the text encoder's embedding dimension. CLIP ViT-L/14
    /// produces 768-dimensional token embeddings, so the U-Net's cross-attention
    /// layers use 768-dimensional keys and values.
    /// </remarks>
    private const int SD15_CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default guidance scale for SD 1.5 (7.5).
    /// </summary>
    /// <remarks>
    /// Classifier-free guidance scale controls how closely the model follows the prompt.
    /// - 1.0: No guidance (unconditional generation)
    /// - 3.0-5.0: Subtle guidance (more creative, less precise)
    /// - 7.0-8.5: Strong guidance (good balance of quality and prompt adherence)
    /// - 10.0+: Very strong guidance (may introduce artifacts)
    ///
    /// 7.5 is the community-standard default for SD 1.5.
    /// </remarks>
    private const double SD15_DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor (865M parameters in the full model).
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding images to latent space and decoding back.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The CLIP text encoder conditioning module.
    /// </summary>
    /// <remarks>
    /// SD 1.5 uses a single CLIP ViT-L/14 text encoder.
    /// This is optional to allow creating the model without a text encoder
    /// for unconditional generation or when the encoder is loaded separately.
    /// </remarks>
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties (ILatentDiffusionModel overrides)

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SD15_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the cross-attention dimension (768 for SD 1.5, matching CLIP ViT-L/14).
    /// </summary>
    public int CrossAttentionDim => SD15_CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableDiffusion15Model with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options for the diffusion model (noise schedule, timesteps, etc.).
    /// If null, uses SD 1.5 defaults: scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with SD 1.5 settings.
    /// DDIM is preferred for SD 1.5 because it allows deterministic generation and
    /// fewer steps (20-50) compared to DDPM's 1000 steps.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net noise predictor. If null, creates the standard SD 1.5 U-Net
    /// with 320 base channels, [1, 2, 4, 4] multipliers, and 768-dim cross-attention
    /// as specified in the original paper.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard SD 1.5 VAE with 128 base channels,
    /// [1, 2, 4, 4] multipliers, 4 latent channels, and 0.18215 scale factor
    /// as specified in the original paper.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically CLIP ViT-L/14).
    /// If null, text-to-image generation will not be available, but unconditional
    /// generation and image-to-image with pre-computed embeddings will still work.
    /// </param>
    /// <param name="seed">
    /// Optional random seed for reproducible generation. Use the same seed with the
    /// same parameters to get identical output images.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults
    /// from the original Stable Diffusion 1.5 paper. You can create a ready-to-use model
    /// with no arguments, or customize any component:
    ///
    /// <code>
    /// // Default configuration (recommended for most users)
    /// var model = new StableDiffusion15Model&lt;float&gt;();
    ///
    /// // With text conditioning for text-to-image
    /// var model = new StableDiffusion15Model&lt;float&gt;(conditioner: myClipEncoder);
    ///
    /// // Full customization
    /// var model = new StableDiffusion15Model&lt;float&gt;(
    ///     unet: myCustomUNet,
    ///     vae: myCustomVAE,
    ///     conditioner: myClipEncoder,
    ///     scheduler: myScheduler,
    ///     seed: 42);
    /// </code>
    /// </para>
    /// </remarks>
    public StableDiffusion15Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
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
        _conditioner = conditioner;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(SD15_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers, using custom layers from the user
    /// if provided or creating industry-standard layers from the SD 1.5 research paper.
    /// </summary>
    /// <param name="unet">
    /// Custom U-Net noise predictor, or null to create the standard SD 1.5 U-Net.
    /// </param>
    /// <param name="vae">
    /// Custom VAE, or null to create the standard SD 1.5 VAE.
    /// </param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// When no custom layers are provided, this method creates the standard SD 1.5 architecture:
    ///
    /// U-Net (865M parameters):
    /// - Input/output channels: 4 (latent space)
    /// - Base channels: 320
    /// - Channel multipliers: [1, 2, 4, 4] producing channels [320, 640, 1280, 1280]
    /// - 2 residual blocks per resolution level
    /// - Self-attention at 8x8, 16x16, and 32x32 latent resolutions
    /// - Cross-attention dimension: 768 (CLIP ViT-L/14 embedding dimension)
    ///
    /// VAE:
    /// - Input: 3-channel RGB images
    /// - Latent: 4 channels
    /// - Base channels: 128
    /// - Channel multipliers: [1, 2, 4, 4] producing channels [128, 256, 512, 512]
    /// - 2 residual blocks per level
    /// - 8x spatial downsampling (512x512 to 64x64)
    /// - Latent scale factor: 0.18215 (normalizes latent variance)
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Priority: 1) User-provided components, 2) Architecture.Layers, 3) Research-paper defaults
        // Use custom U-Net from user, or create with Architecture layers, or use SD 1.5 defaults
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: SD15_LATENT_CHANNELS,
            outputChannels: SD15_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: SD15_CROSS_ATTENTION_DIM,
            seed: seed);

        // Use custom VAE from user, or create industry-standard SD 1.5 VAE
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD15_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates an image from a text prompt using SD 1.5 defaults.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">
    /// Optional negative prompt describing what to avoid. Common negative prompts for SD 1.5:
    /// "blurry, bad anatomy, worst quality, low quality, watermark, text"
    /// </param>
    /// <param name="width">
    /// Image width in pixels. Must be divisible by 8 (VAE requirement).
    /// Default: 512. Best results at 512x512 (native training resolution).
    /// </param>
    /// <param name="height">
    /// Image height in pixels. Must be divisible by 8.
    /// Default: 512.
    /// </param>
    /// <param name="numInferenceSteps">
    /// Number of denoising steps. More steps = higher quality but slower.
    /// - 20-30 steps: Fast, good quality
    /// - 50 steps: High quality (default)
    /// - 100+ steps: Diminishing returns
    /// </param>
    /// <param name="guidanceScale">
    /// Classifier-free guidance scale. If null, uses 7.5 (SD 1.5 default).
    /// See <see cref="SD15_DEFAULT_GUIDANCE_SCALE"/> for recommended range.
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>
    /// Generated image tensor with shape [1, 3, height, width] in [-1, 1] range.
    /// To convert to display format: pixel = (value + 1) * 127.5
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for generating images from text.
    ///
    /// Tips for good SD 1.5 prompts:
    /// - Be descriptive: "A detailed oil painting of a castle on a cliff at sunset"
    /// - Include quality tags: "masterpiece, best quality, highly detailed"
    /// - Specify style: "photorealistic", "anime style", "watercolor painting"
    /// - Use negative prompts to avoid common issues
    ///
    /// The generation pipeline:
    /// 1. Text prompt -> CLIP ViT-L/14 -> 77x768 embedding
    /// 2. Random noise -> 1x4x64x64 latent
    /// 3. U-Net denoises latent with text guidance (50 steps)
    /// 4. VAE decodes latent -> 1x3x512x512 image
    /// </para>
    /// </remarks>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SD15_DEFAULT_GUIDANCE_SCALE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Performs image-to-image transformation using SD 1.5.
    /// </summary>
    /// <param name="inputImage">The source image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">Text prompt describing the desired transformation.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// How much to transform the input (0.0 = no change, 1.0 = full regeneration).
    /// - 0.3-0.5: Subtle changes (color grading, minor edits)
    /// - 0.6-0.8: Major changes (style transfer, significant edits)
    /// - 0.9-1.0: Near-complete regeneration
    /// </param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Transformed image tensor.</returns>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.75,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SD15_DEFAULT_GUIDANCE_SCALE;

        return base.ImageToImage(
            inputImage,
            prompt,
            negativePrompt,
            strength,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Generates multiple image variations from the same prompt using different seeds.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired images.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="count">Number of variations to generate (default: 4).</param>
    /// <param name="width">Image width (default: 512).</param>
    /// <param name="height">Image height (default: 512).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="baseSeed">
    /// Base seed for variations. Variation i uses seed = baseSeed + i.
    /// If null, a random base seed is chosen.
    /// </param>
    /// <returns>List of generated image tensors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This generates multiple different images from the same prompt.
    /// Each variation uses a different random seed, producing different compositions
    /// while following the same text description.
    /// </para>
    /// </remarks>
    public virtual List<Tensor<T>> GenerateVariations(
        string prompt,
        string? negativePrompt = null,
        int count = 4,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? baseSeed = null)
    {
        var results = new List<Tensor<T>>();
        var startSeed = baseSeed ?? RandomGenerator.Next();

        for (int i = 0; i < count; i++)
        {
            var image = GenerateFromText(
                prompt,
                negativePrompt,
                width,
                height,
                numInferenceSteps,
                guidanceScale,
                startSeed + i);
            results.Add(image);
        }

        return results;
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
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
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

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
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: SD15_LATENT_CHANNELS,
            outputChannels: SD15_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: SD15_CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD15_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new StableDiffusion15Model<T>(
            unet: clonedUnet,
            vae: clonedVae,
            conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Stable Diffusion 1.5",
            Version = "1.5",
            ModelType = ModelType.NeuralNetwork,
            Description = "Stable Diffusion 1.5 latent diffusion model with CLIP ViT-L/14 text conditioning",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "latent-diffusion");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("cross_attention_dim", SD15_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", SD15_LATENT_CHANNELS);
        metadata.SetProperty("vae_scale_factor", SD15_VAE_SCALE_FACTOR);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("latent_scale", 0.18215);
        metadata.SetProperty("beta_schedule", "scaled_linear");

        return metadata;
    }

    #endregion
}
