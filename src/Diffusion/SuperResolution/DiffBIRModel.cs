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
/// DiffBIR model for blind image restoration with generative diffusion prior.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiffBIR (Diffusion-Based Blind Image Restoration) uses a two-stage pipeline for
/// real-world image restoration. The first stage removes degradation using a regression
/// module (SwinIR), and the second stage refines details using a Stable Diffusion prior
/// with a controllable module for balancing fidelity and quality.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Stage 1: SwinIR regression module for degradation removal</description></item>
/// <item><description>Stage 2: SD 1.5 U-Net backbone (320 base channels, [1,2,4,4] multipliers)</description></item>
/// <item><description>Controllable feature wrapping module (LAControlNet) for fidelity-quality balance</description></item>
/// <item><description>Standard VAE (4-channel latent space, 0.18215 scale factor)</description></item>
/// <item><description>CLIP ViT-L/14 text encoder (768-dim cross-attention)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DiffBIR restores degraded images (blur, noise, JPEG artifacts, old photos).
///
/// How DiffBIR works:
/// 1. Stage 1 (SwinIR): A regression network removes obvious degradation from the input
/// 2. Stage 2 (Diffusion): The SD prior adds realistic details to the cleaned image
/// 3. LAControlNet: Controls how much the diffusion stage can modify the Stage 1 output
///
/// Key characteristics:
/// - Handles blind (unknown) degradation types automatically
/// - Two-stage design separates degradation removal from detail generation
/// - Controllable balance between fidelity to input and generated quality
/// - Works for face restoration, general images, and old photo restoration
/// - Based on Stable Diffusion 1.5 backbone for high-quality detail generation
///
/// When to use DiffBIR:
/// - Restoring old or damaged photographs
/// - Removing JPEG compression artifacts
/// - Denoising and deblurring real-world images
/// - Face restoration in group photos or surveillance footage
///
/// Limitations:
/// - Two-stage pipeline is slower than single-stage methods
/// - May hallucinate details not present in the original image
/// - Best results at 512x512 (SD 1.5 native resolution)
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Two-stage (SwinIR + SD 1.5 U-Net with LAControlNet)
/// - Stage 1: SwinIR with 6 RSTB blocks, 180 channels, window size 8
/// - Stage 2: SD 1.5 U-Net backbone, 320 base channels, [1,2,4,4] multipliers
/// - Cross-attention dimension: 768 (CLIP ViT-L/14)
/// - VAE: 4 latent channels, scale factor 0.18215, 8x downsampling
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Default guidance scale: 7.5
/// - Default resolution: 512x512
///
/// Reference: Lin et al., "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior", ECCV 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var diffbir = new DiffBIRModel&lt;float&gt;();
///
/// // Restore a degraded image using image-to-image pipeline
/// var restored = diffbir.ImageToImage(
///     inputImage: degradedPhoto,
///     prompt: "high quality, sharp, detailed",
///     negativePrompt: "blurry, noisy, artifacts",
///     strength: 0.5,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate from text (for unconditional restoration with prompt guidance)
/// var result = diffbir.GenerateFromText(
///     prompt: "a sharp, detailed photograph",
///     width: 512,
///     height: 512,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class DiffBIRModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for DiffBIR restoration.
    /// </summary>
    /// <remarks>
    /// DiffBIR uses the SD 1.5 backbone which was trained on 512x512 images.
    /// Input images are tiled at this resolution for processing.
    /// </remarks>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for DiffBIR restoration.
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels in the standard SD VAE (4).
    /// </summary>
    /// <remarks>
    /// The VAE compresses 3-channel RGB images into a 4-channel latent representation.
    /// This matches the SD 1.5 VAE used as DiffBIR's latent encoder.
    /// </remarks>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-L/14 output (768).
    /// </summary>
    /// <remarks>
    /// DiffBIR uses the SD 1.5 backbone which conditions on CLIP ViT-L/14
    /// text embeddings with 768-dimensional token vectors.
    /// </remarks>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default classifier-free guidance scale for DiffBIR (7.5).
    /// </summary>
    /// <remarks>
    /// Controls how closely the diffusion stage follows the text prompt.
    /// For restoration, moderate guidance (5.0-7.5) preserves input structure
    /// while adding realistic details.
    /// </remarks>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Base channel count for the SD 1.5 U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// VAE spatial downsampling factor (8x).
    /// </summary>
    private const int VAE_SCALE_FACTOR = 8;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor using the SD 1.5 backbone architecture.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The standard VAE for encoding images to latent space and decoding back.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The CLIP text encoder conditioning module for prompt-guided restoration.
    /// </summary>
    /// <remarks>
    /// Optional to allow creating the model without a text encoder for
    /// unconditional restoration or when embeddings are pre-computed.
    /// </remarks>
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
    /// Gets the cross-attention dimension (768, matching CLIP ViT-L/14).
    /// </summary>
    public int CrossAttentionDim => CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of DiffBIRModel with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture for custom layer configuration.
    /// If provided, custom layers will be passed to the U-Net noise predictor.
    /// </param>
    /// <param name="options">
    /// Diffusion model options (noise schedule, timesteps, etc.).
    /// If null, uses DiffBIR defaults: scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses a DDPM scheduler with Stable Diffusion settings,
    /// matching the original DiffBIR training configuration.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net noise predictor. If null, creates the standard SD 1.5 U-Net
    /// backbone with 320 base channels and 768-dim cross-attention.
    /// </param>
    /// <param name="vae">
    /// Custom standard VAE. If null, creates the standard SD 1.5 VAE with
    /// 4 latent channels and 0.18215 scale factor.
    /// </param>
    /// <param name="conditioner">
    /// Text conditioning module (typically CLIP ViT-L/14).
    /// If null, prompt-guided restoration will not be available.
    /// </param>
    /// <param name="seed">Optional random seed for reproducible restoration.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults.
    /// Create a ready-to-use model with no arguments:
    ///
    /// <code>
    /// // Default configuration
    /// var model = new DiffBIRModel&lt;float&gt;();
    ///
    /// // With text conditioning for prompt-guided restoration
    /// var model = new DiffBIRModel&lt;float&gt;(conditioner: myClipEncoder);
    /// </code>
    /// </para>
    /// </remarks>
    public DiffBIRModel(
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
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers, using custom components if provided
    /// or creating industry-standard layers from the DiffBIR research paper.
    /// </summary>
    /// <param name="unet">Custom U-Net noise predictor, or null to create the default.</param>
    /// <param name="vae">Custom VAE, or null to create the default.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// When no custom layers are provided, this method creates the SD 1.5 backbone:
    ///
    /// U-Net:
    /// - Input/output channels: 4 (latent space)
    /// - Base channels: 320
    /// - Channel multipliers: [1, 2, 4, 4] producing channels [320, 640, 1280, 1280]
    /// - 2 residual blocks per resolution level
    /// - Cross-attention dimension: 768 (CLIP ViT-L/14)
    /// - Attention at 4x, 2x, and 1x downsampling levels
    ///
    /// VAE:
    /// - Input: 3-channel RGB
    /// - Latent: 4 channels
    /// - Base channels: 128, multipliers [1, 2, 4, 4]
    /// - Scale factor: 0.18215
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
            inputChannels: LATENT_CHANNELS,
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
    /// Generates an image from a text prompt using DiffBIR defaults.
    /// </summary>
    /// <param name="prompt">Text prompt describing the desired restoration quality.</param>
    /// <param name="negativePrompt">
    /// Optional negative prompt. Common for DiffBIR:
    /// "blurry, noisy, artifacts, low quality, jpeg artifacts"
    /// </param>
    /// <param name="width">Image width (default: 512, must be divisible by 8).</param>
    /// <param name="height">Image height (default: 512, must be divisible by 8).</param>
    /// <param name="numInferenceSteps">
    /// Number of denoising steps. DiffBIR typically uses 50 steps for high quality.
    /// </param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>Generated image tensor with shape [1, 3, height, width].</returns>
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
    /// Performs image restoration using the DiffBIR diffusion pipeline.
    /// </summary>
    /// <param name="inputImage">The degraded image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">
    /// Text prompt describing the desired restoration. For DiffBIR, prompts like
    /// "high quality, sharp, detailed photograph" work well.
    /// </param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// Controls how much the diffusion stage modifies the input (0.0-1.0).
    /// For restoration, lower values (0.3-0.5) preserve more of the original structure.
    /// - 0.3-0.4: Conservative restoration (preserves input faithfully)
    /// - 0.5-0.6: Balanced restoration (good fidelity-quality trade-off)
    /// - 0.7-0.8: Aggressive restoration (may hallucinate details)
    /// </param>
    /// <param name="numInferenceSteps">Number of denoising steps (default: 50).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Restored image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the primary method for DiffBIR image restoration.
    /// Feed in a degraded image and get back a restored version with enhanced details.
    /// The strength parameter controls the fidelity-quality trade-off.
    /// </para>
    /// </remarks>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
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
            inputChannels: LATENT_CHANNELS,
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

        return new DiffBIRModel<T>(
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
            Name = "DiffBIR",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "DiffBIR blind image restoration with generative diffusion prior",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "two-stage-diffusion-restoration");
        metadata.SetProperty("stage1", "SwinIR regression");
        metadata.SetProperty("stage2", "SD 1.5 U-Net backbone");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("vae_scale_factor", VAE_SCALE_FACTOR);
        metadata.SetProperty("tasks", "blind-restoration,face-restoration,old-photo,denoising");
        metadata.SetProperty("beta_schedule", "scaled_linear");
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
