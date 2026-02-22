using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// Stable Diffusion x4 Upscaler model for text-guided latent super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Stable Diffusion x4 Upscaler takes a low-resolution image and upscales it 4x
/// using a latent diffusion process conditioned on both the low-resolution input
/// (concatenated in latent space) and a text prompt for guided detail generation.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>U-Net with 7 input channels (4 latent + 3 low-res RGB conditioning)</description></item>
/// <item><description>320 base channels with [1, 2, 4, 4] multipliers (SD 2.x backbone)</description></item>
/// <item><description>OpenCLIP ViT-H/14 text encoder (1024-dim cross-attention)</description></item>
/// <item><description>Standard VAE (4-channel latent space, 0.18215 scale factor)</description></item>
/// <item><description>DDIM scheduler for faster inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This model increases image resolution by 4x with text guidance.
///
/// How the SD Upscaler works:
/// 1. The low-resolution image is resized and concatenated with latent noise (7 input channels)
/// 2. A text prompt guides what details to add during upscaling
/// 3. The U-Net denoises in latent space over 75 steps
/// 4. The VAE decodes the result to a 4x larger image
///
/// Key characteristics:
/// - 4x upscaling (128 to 512, 256 to 1024, etc.)
/// - Text-guided: prompts control what details are added
/// - 7 input channels: 4 latent + 3 low-res RGB conditioning
/// - Based on SD 2.x architecture with OpenCLIP text encoder
/// - DDIM scheduler for efficient 75-step inference
///
/// When to use SD Upscaler:
/// - Upscaling images with specific desired detail types
/// - Combining upscaling with style transfer
/// - When you want prompt control over the upscaling process
/// - AI-generated image enhancement
///
/// Limitations:
/// - Fixed 4x upscale factor
/// - Slower than Real-ESRGAN due to 75 inference steps
/// - May add details inconsistent with original content at high guidance
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: U-Net with low-res RGB conditioning
/// - Input channels: 7 (4 latent noise + 3 low-res RGB)
/// - Output channels: 4 (latent space)
/// - Base channels: 320, multipliers [1, 2, 4, 4]
/// - Cross-attention dimension: 1024 (OpenCLIP ViT-H/14)
/// - 2 residual blocks per level
/// - Attention at 4x, 2x, and 1x downsampling
/// - Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
/// - Default inference steps: 75
/// - Default guidance scale: 7.5
/// - Upscale factor: 4x
///
/// Reference: Rombach et al., "Stable Diffusion x4 Upscaler", Stability AI, 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var upscaler = new SDUpscalerModel&lt;float&gt;();
///
/// // Text-guided 4x upscaling
/// var upscaled = upscaler.ImageToImage(
///     inputImage: lowResImage,
///     prompt: "high resolution, sharp details, 4k photograph",
///     negativePrompt: "blurry, pixelated, noisy",
///     strength: 0.75,
///     numInferenceSteps: 75,
///     guidanceScale: 7.5);
///
/// // Simple generation for testing
/// var result = upscaler.GenerateFromText(
///     prompt: "a detailed landscape photograph",
///     width: 512,
///     height: 512);
/// </code>
/// </example>
public class SDUpscalerModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default output image width (512).
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default output image height (512).
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Total input channels for the U-Net (7 = 4 latent + 3 low-res RGB).
    /// </summary>
    /// <remarks>
    /// The U-Net receives concatenated latent noise (4 channels) and the
    /// downscaled low-resolution input image (3 RGB channels).
    /// </remarks>
    private const int INPUT_CHANNELS = 7;

    /// <summary>
    /// Cross-attention dimension matching OpenCLIP ViT-H/14 output (1024).
    /// </summary>
    /// <remarks>
    /// The SD x4 Upscaler uses the SD 2.x backbone which conditions on OpenCLIP
    /// text embeddings with 1024-dimensional token vectors.
    /// </remarks>
    private const int CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Default classifier-free guidance scale (7.5).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Upscale factor (4x).
    /// </summary>
    private const int UPSCALE_FACTOR = 4;

    /// <summary>
    /// Base channel count for the U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor with low-resolution RGB conditioning.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The standard VAE for encoding/decoding between pixel and latent space.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The OpenCLIP text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the upscale factor (4x).
    /// </summary>
    public int UpscaleFactor => UPSCALE_FACTOR;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of SDUpscalerModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses SD Upscaler defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDIM with SD settings for efficient inference.
    /// </param>
    /// <param name="unet">Custom U-Net. If null, creates the standard 7-input-channel U-Net.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD VAE.</param>
    /// <param name="conditioner">Optional text conditioning module (OpenCLIP ViT-H/14).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SDUpscalerModel(
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
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="unet">Custom U-Net, or null for SD Upscaler defaults.</param>
    /// <param name="vae">Custom VAE, or null for standard VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default U-Net architecture:
    /// - Input: 7 channels (4 latent + 3 low-res RGB)
    /// - Output: 4 channels (latent space)
    /// - Base channels: 320, multipliers [1, 2, 4, 4]
    /// - 1024-dim cross-attention (OpenCLIP ViT-H/14)
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: INPUT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates an image from a text prompt using SD Upscaler defaults.
    /// </summary>
    /// <param name="prompt">Text prompt describing desired output.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width (default: 512).</param>
    /// <param name="height">Output height (default: 512).</param>
    /// <param name="numInferenceSteps">Denoising steps (default: 75 for upscaling quality).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated image tensor.</returns>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 75,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? DEFAULT_GUIDANCE_SCALE;

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
    /// Performs text-guided 4x image upscaling.
    /// </summary>
    /// <param name="inputImage">The low-resolution image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">Text prompt guiding what details to add during upscaling.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// Upscaling strength (0.0-1.0):
    /// - 0.5-0.7: Faithful upscaling with moderate detail addition
    /// - 0.7-0.9: More creative detail generation
    /// </param>
    /// <param name="numInferenceSteps">Number of denoising steps (default: 75).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Upscaled image tensor at 4x resolution.</returns>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.75,
        int numInferenceSteps = 75,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? DEFAULT_GUIDANCE_SCALE;

        return base.ImageToImage(
            inputImage,
            prompt,
            negativePrompt,
            strength,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);

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
            inputChannels: INPUT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new SDUpscalerModel<T>(
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
            Name = "SD-Upscaler-x4",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Stable Diffusion x4 Upscaler for text-guided latent super-resolution",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sd-upscaler");
        metadata.SetProperty("text_encoder", "OpenCLIP ViT-H/14");
        metadata.SetProperty("upscale_factor", UPSCALE_FACTOR);
        metadata.SetProperty("input_channels", INPUT_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("noise_schedule", "linear");
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "DDIM");

        return metadata;
    }

    #endregion
}
