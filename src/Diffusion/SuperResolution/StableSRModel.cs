using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// StableSR model for exploiting diffusion prior for real-world image super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StableSR leverages the generative prior of a pretrained Stable Diffusion model for
/// real-world image super-resolution. It introduces a controllable feature wrapping (CFW)
/// module and a time-aware encoder that adapts features to the diffusion timestep for
/// balancing fidelity to the input and perceptual quality.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Pretrained SD 1.5 U-Net backbone (frozen during SR training)</description></item>
/// <item><description>Controllable Feature Wrapping (CFW) module for fidelity-quality balance</description></item>
/// <item><description>Time-aware encoder that adapts to diffusion timestep</description></item>
/// <item><description>Standard SD VAE (4-channel latent space, 0.18215 scale factor)</description></item>
/// <item><description>CLIP ViT-L/14 text encoder (768-dim cross-attention)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> StableSR uses Stable Diffusion's knowledge for image upscaling.
///
/// How StableSR works:
/// 1. The degraded image is encoded to latent space by a time-aware encoder
/// 2. The CFW module wraps features from the encoder and injects them into the frozen SD U-Net
/// 3. The U-Net generates realistic details guided by these features
/// 4. A controllable strength parameter balances fidelity vs quality
/// 5. More diffusion steps (200 default) produce higher quality
///
/// Key characteristics:
/// - Leverages pretrained SD 1.5 prior for realistic detail generation
/// - Controllable fidelity-quality trade-off via CFW module
/// - Time-aware encoder adapts to diffusion timestep for better results
/// - Better perceptual quality than pure regression methods (PSNR-focused)
/// - 200 inference steps by default (more than standard diffusion models)
///
/// When to use StableSR:
/// - Upscaling photographs where perceptual quality matters more than pixel accuracy
/// - Restoring images where you want natural-looking details
/// - When you need fine control over fidelity-quality balance
/// - Combining with SD ecosystem tools (ControlNet, LoRA)
///
/// Limitations:
/// - Slower than GAN-based SR methods (200 inference steps)
/// - May generate details not present in the original image
/// - Requires more VRAM than lightweight SR models
/// - Best at 512x512 output (SD 1.5 native resolution)
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net + CFW module + time-aware encoder
/// - U-Net backbone: 320 base channels, [1, 2, 4, 4] multipliers (frozen)
/// - Cross-attention dimension: 768 (CLIP ViT-L/14)
/// - CFW: Controllable feature wrapping for each U-Net decoder block
/// - VAE: 4 latent channels, scale factor 0.18215
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Default inference steps: 200 (higher than standard for SR quality)
/// - Default guidance scale: 7.5
///
/// Reference: Wang et al., "Exploiting Diffusion Prior for Real-World Image Super-Resolution", IJCV 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var stableSR = new StableSRModel&lt;float&gt;();
///
/// // Upscale with default 200 steps for maximum quality
/// var upscaled = stableSR.ImageToImage(
///     inputImage: lowResPhoto,
///     prompt: "high quality, detailed, sharp photograph",
///     negativePrompt: "blurry, noisy, artifacts, low quality",
///     strength: 0.5,
///     numInferenceSteps: 200,
///     guidanceScale: 7.5);
///
/// // Quick upscale with fewer steps
/// var quickResult = stableSR.ImageToImage(
///     inputImage: lowResPhoto,
///     prompt: "detailed photograph",
///     strength: 0.4,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class StableSRModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for StableSR output.
    /// </summary>
    /// <remarks>
    /// StableSR uses the SD 1.5 backbone trained on 512x512 images.
    /// </remarks>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for StableSR output.
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels in the SD VAE (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-L/14 output (768).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default guidance scale for StableSR (7.5).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Base channel count for the SD 1.5 U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// Default number of inference steps for StableSR (200).
    /// </summary>
    /// <remarks>
    /// StableSR uses more steps than standard diffusion models because the
    /// restoration task benefits from gradual detail refinement over many steps.
    /// </remarks>
    private const int DEFAULT_INFERENCE_STEPS = 200;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor using the frozen SD 1.5 backbone.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The standard VAE for encoding/decoding between pixel and latent space.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// Optional CLIP text encoder conditioning module.
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

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableSRModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses StableSR defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDPM with SD settings matching the StableSR paper.
    /// </param>
    /// <param name="unet">Custom U-Net. If null, creates the standard SD 1.5 U-Net backbone.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD VAE.</param>
    /// <param name="conditioner">Optional text conditioning module (CLIP ViT-L/14).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StableSRModel(
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
    /// Initializes the U-Net and VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="unet">Custom U-Net, or null for SD 1.5 defaults.</param>
    /// <param name="vae">Custom VAE, or null for standard VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default architecture matches the frozen SD 1.5 backbone:
    /// - U-Net: 320 base channels, [1, 2, 4, 4] multipliers, 768-dim cross-attention
    /// - VAE: 4 latent channels, 128 base channels, 0.18215 scale factor
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
    /// Generates an image from a text prompt using StableSR defaults.
    /// </summary>
    /// <param name="prompt">Text prompt.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width (default: 512).</param>
    /// <param name="height">Output height (default: 512).</param>
    /// <param name="numInferenceSteps">
    /// Denoising steps (default: 200). StableSR benefits from more steps than standard models.
    /// </param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated image tensor.</returns>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = DEFAULT_INFERENCE_STEPS,
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
    /// Performs diffusion-prior-based image super-resolution.
    /// </summary>
    /// <param name="inputImage">The low-resolution image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">
    /// Text prompt guiding the restoration. For StableSR, prompts like
    /// "high quality, sharp, detailed" improve results.
    /// </param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// Controls how much the diffusion prior modifies the input (0.0-1.0):
    /// - 0.3-0.4: Conservative (high fidelity, less detail)
    /// - 0.5-0.6: Balanced (recommended default)
    /// - 0.7-0.8: Aggressive (more hallucinated detail)
    /// </param>
    /// <param name="numInferenceSteps">Denoising steps (default: 200).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Super-resolved image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> StableSR uses 200 steps by default (more than most models)
    /// because gradual refinement produces better restoration results. Use fewer steps
    /// (50-100) for faster but lower quality results.
    /// </para>
    /// </remarks>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = DEFAULT_INFERENCE_STEPS,
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

        return new StableSRModel<T>(
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
            Name = "StableSR",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "StableSR diffusion-prior-based super-resolution with controllable feature wrapping",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sd15-controllable-sr");
        metadata.SetProperty("backbone", "SD 1.5 U-Net (frozen)");
        metadata.SetProperty("feature_wrapping", true);
        metadata.SetProperty("time_aware_encoder", true);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("default_inference_steps", DEFAULT_INFERENCE_STEPS);
        metadata.SetProperty("beta_schedule", "scaled_linear");
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
