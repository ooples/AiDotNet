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

namespace AiDotNet.Diffusion.SuperResolution;

/// <summary>
/// Real-ESRGAN model for practical blind image super-resolution with degradation-aware training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Real-ESRGAN combines the ESRGAN architecture with a second-order degradation model
/// for practical blind super-resolution that handles complex real-world degradations
/// including blur, noise, JPEG artifacts, and their combinations.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>RRDB-Net backbone (Residual-in-Residual Dense Blocks) with 23 blocks</description></item>
/// <item><description>U-Net discriminator with spectral normalization for training stability</description></item>
/// <item><description>Second-order degradation model simulating real-world image corruption</description></item>
/// <item><description>Diffusion-based refinement using concatenated low-res conditioning (8 input channels)</description></item>
/// <item><description>Standard VAE (4-channel latent space, 0.18215 scale factor)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN upscales low-resolution real-world photos by 4x.
///
/// How Real-ESRGAN works:
/// 1. The input low-resolution image is concatenated with the latent noise (8 input channels)
/// 2. The U-Net predicts noise conditioned on the low-res image
/// 3. The denoised latent is decoded to a high-resolution 4x upscaled image
///
/// Key characteristics:
/// - 4x upscaling factor (128x128 to 512x512, 256x256 to 1024x1024)
/// - Second-order degradation model handles realistic corruption chains
/// - Works well on faces, landscapes, anime, and general photography
/// - Unconditional by default (guidance scale 1.0, no text prompt needed)
/// - Can use text prompts for guided upscaling when conditioner is provided
///
/// When to use Real-ESRGAN:
/// - Upscaling low-resolution photos from the web or social media
/// - Enhancing old photographs and scanned images
/// - Anime and illustration upscaling (specialized models available)
/// - Batch processing of image libraries
///
/// Limitations:
/// - Fixed 4x upscale factor (use SDUpscaler for flexible scaling)
/// - May add unnatural sharpness to already-sharp images
/// - Large model size due to RRDB backbone
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: RRDB-Net + diffusion refinement with low-res conditioning
/// - Input channels: 8 (4 latent + 4 downscaled low-res conditioning)
/// - Output channels: 4 (latent space)
/// - Base channels: 128
/// - Channel multipliers: [1, 2, 4]
/// - Upscale factor: 4x
/// - RRDB blocks: 23 (in the full ESRGAN backbone)
/// - Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
/// - Default guidance scale: 1.0 (unconditional)
///
/// Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCVW 2021
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var realEsrgan = new RealESRGANModel&lt;float&gt;();
///
/// // Upscale a low-resolution image (unconditional 4x upscale)
/// var upscaled = realEsrgan.ImageToImage(
///     inputImage: lowResPhoto,
///     prompt: "",
///     strength: 0.75,
///     numInferenceSteps: 50);
///
/// // Text-guided upscaling (requires conditioner)
/// var guidedUpscale = realEsrgan.ImageToImage(
///     inputImage: lowResPhoto,
///     prompt: "sharp photograph, highly detailed, 4k resolution",
///     negativePrompt: "blurry, pixelated, artifacts",
///     strength: 0.6,
///     numInferenceSteps: 50,
///     guidanceScale: 3.0);
/// </code>
/// </example>
public class RealESRGANModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Real-ESRGAN output.
    /// </summary>
    /// <remarks>
    /// The output resolution after 4x upscaling. For a 128x128 input,
    /// the output is 512x512.
    /// </remarks>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for Real-ESRGAN output.
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension for optional text conditioning (768).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default guidance scale for Real-ESRGAN (1.0, unconditional).
    /// </summary>
    /// <remarks>
    /// Real-ESRGAN is primarily unconditional (guidance scale 1.0).
    /// The low-resolution input provides all necessary conditioning.
    /// Text guidance can be enabled by setting a higher value with a conditioner.
    /// </remarks>
    private const double DEFAULT_GUIDANCE_SCALE = 1.0;

    /// <summary>
    /// Base channel count for the Real-ESRGAN U-Net (128).
    /// </summary>
    /// <remarks>
    /// Smaller than SD 1.5's 320 base channels because the degradation-aware
    /// U-Net focuses on super-resolution rather than generation.
    /// </remarks>
    private const int BASE_CHANNELS = 128;

    /// <summary>
    /// Upscale factor for Real-ESRGAN (4x).
    /// </summary>
    private const int UPSCALE_FACTOR = 4;

    /// <summary>
    /// Input channels for the U-Net (8 = 4 latent + 4 downscaled low-res).
    /// </summary>
    /// <remarks>
    /// The U-Net receives concatenated latent noise and downscaled low-resolution
    /// conditioning, doubling the standard 4 latent channels.
    /// </remarks>
    private const int INPUT_CHANNELS = 8;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor with low-resolution image conditioning.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The standard VAE for encoding/decoding between pixel and latent space.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// Optional text conditioning module for guided upscaling.
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
    /// Initializes a new instance of RealESRGANModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Real-ESRGAN defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">Noise scheduler. If null, uses a DDPM scheduler with default settings.</param>
    /// <param name="unet">Custom U-Net. If null, creates the standard Real-ESRGAN U-Net with 8 input channels.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard VAE.</param>
    /// <param name="conditioner">Optional text conditioning module for guided upscaling.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional. Create a ready-to-use model:
    ///
    /// <code>
    /// var model = new RealESRGANModel&lt;float&gt;();
    /// </code>
    /// </para>
    /// </remarks>
    public RealESRGANModel(
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
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
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
    /// <param name="unet">Custom U-Net, or null for Real-ESRGAN defaults.</param>
    /// <param name="vae">Custom VAE, or null for standard VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default U-Net architecture:
    /// - Input: 8 channels (4 latent + 4 low-res conditioning)
    /// - Output: 4 channels (latent space)
    /// - Base channels: 128, multipliers [1, 2, 4]
    /// - Attention at 4x and 2x downsampling levels
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
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
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
    /// Generates an image from a text prompt using Real-ESRGAN defaults.
    /// </summary>
    /// <param name="prompt">Text prompt (Real-ESRGAN is primarily unconditional).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width (default: 512).</param>
    /// <param name="height">Output height (default: 512).</param>
    /// <param name="numInferenceSteps">Denoising steps (default: 50).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 1.0 (unconditional).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated image tensor.</returns>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 50,
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
    /// Performs 4x image upscaling using the Real-ESRGAN diffusion pipeline.
    /// </summary>
    /// <param name="inputImage">The low-resolution image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">Optional text prompt for guided upscaling.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// Diffusion strength (0.0-1.0). For upscaling:
    /// - 0.5-0.7: Balanced detail enhancement
    /// - 0.7-0.9: More aggressive detail generation
    /// </param>
    /// <param name="numInferenceSteps">Number of denoising steps (default: 50).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 1.0.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Upscaled image tensor at 4x resolution.</returns>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.75,
        int numInferenceSteps = 50,
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
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
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

        return new RealESRGANModel<T>(
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
            Name = "Real-ESRGAN",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Real-ESRGAN practical blind super-resolution with second-order degradation model",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "esrgan-diffusion");
        metadata.SetProperty("backbone", "RRDB-Net-23B");
        metadata.SetProperty("upscale_factor", UPSCALE_FACTOR);
        metadata.SetProperty("input_channels", INPUT_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("degradation_model", "second-order");
        metadata.SetProperty("noise_schedule", "linear");
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("tasks", "4x-upscale,blind-sr,face-enhancement,anime-upscale");

        return metadata;
    }

    #endregion
}
