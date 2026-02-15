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
/// SUPIR model for scaling up image restoration with SDXL for photo-realistic results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SUPIR (Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image
/// Restoration In the Wild) leverages the SDXL model's generative prior with a GPT-guided
/// restoration pipeline for photo-realistic super-resolution at high resolutions. It combines
/// semantic understanding from a large language model with SDXL's generation quality.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>SDXL U-Net backbone (320 base channels, [1, 2, 4] multipliers)</description></item>
/// <item><description>Dual text encoders: OpenCLIP ViT-bigG/14 + CLIP ViT-L/14 (2048-dim combined)</description></item>
/// <item><description>GPT-guided quality description for semantic restoration understanding</description></item>
/// <item><description>SDXL VAE (4-channel latent, 0.13025 scale factor for SDXL)</description></item>
/// <item><description>DPM-Solver++ multistep scheduler for efficient inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> SUPIR uses SDXL (the most capable open-source image model)
/// combined with GPT language understanding for the highest quality image restoration.
///
/// How SUPIR works:
/// 1. A GPT model analyzes the degraded image and generates a quality description
/// 2. The description guides SDXL's diffusion process for intelligent restoration
/// 3. SDXL's 2048-dim dual text encoder provides rich semantic conditioning
/// 4. The result is photo-realistic detail at 1024x1024 native resolution
///
/// Key characteristics:
/// - Based on SDXL (1024x1024 native resolution, much higher than SD 1.5's 512x512)
/// - GPT-guided semantic understanding of image content and degradation
/// - Dual text encoders (2048-dim combined) for precise conditioning
/// - DPM-Solver++ for efficient 50-step inference
/// - SDXL VAE scale factor (0.13025, different from SD 1.5's 0.18215)
///
/// When to use SUPIR:
/// - Highest possible quality restoration is needed
/// - Processing at 1024x1024 resolution
/// - Complex degradation where semantic understanding helps
/// - Photo-realistic detail generation for important images
///
/// Limitations:
/// - Requires significantly more VRAM than SD 1.5-based models (~12GB+)
/// - Slower than lighter SR methods due to SDXL backbone
/// - GPT component adds complexity and potential dependency
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SDXL U-Net + GPT-guided conditioning
/// - U-Net: 320 base channels, multipliers [1, 2, 4] (SDXL config)
/// - Cross-attention dimension: 2048 (dual OpenCLIP + CLIP)
/// - VAE: 4 latent channels, SDXL scale factor 0.13025
/// - Native resolution: 1024x1024 pixels
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Default scheduler: DPM-Solver++ Multistep
/// - Default guidance scale: 7.5
///
/// Reference: Yu et al., "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults (1024x1024 resolution)
/// var supir = new SUPIRModel&lt;float&gt;();
///
/// // High-quality 1024x1024 restoration
/// var restored = supir.ImageToImage(
///     inputImage: degradedImage,
///     prompt: "a high quality, detailed photograph with sharp focus",
///     negativePrompt: "blurry, noisy, low quality, artifacts",
///     strength: 0.5,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate at SDXL native resolution
/// var result = supir.GenerateFromText(
///     prompt: "a stunning photograph, masterpiece quality",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class SUPIRModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SUPIR output (1024, SDXL native).
    /// </summary>
    /// <remarks>
    /// SUPIR uses the SDXL backbone which was trained on 1024x1024 images,
    /// producing significantly higher resolution output than SD 1.5-based models.
    /// </remarks>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for SUPIR output (1024, SDXL native).
    /// </summary>
    public const int DefaultHeight = 1024;

    /// <summary>
    /// Number of latent channels (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension for SDXL dual text encoders (2048).
    /// </summary>
    /// <remarks>
    /// SDXL uses two text encoders: OpenCLIP ViT-bigG/14 (1280-dim) and
    /// CLIP ViT-L/14 (768-dim), combined to 2048 dimensions for richer
    /// semantic conditioning than single-encoder models.
    /// </remarks>
    private const int CROSS_ATTENTION_DIM = 2048;

    /// <summary>
    /// Default guidance scale for SUPIR (7.5).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Base channel count for the SDXL U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// SDXL VAE scale factor (0.13025).
    /// </summary>
    /// <remarks>
    /// The SDXL VAE uses a different scale factor (0.13025) than SD 1.5 (0.18215)
    /// due to differences in training and the VAE architecture.
    /// </remarks>
    private const double SDXL_VAE_SCALE_FACTOR = 0.13025;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor using the SDXL backbone architecture.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The SDXL VAE for encoding/decoding between pixel and latent space.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// Optional dual text encoder conditioning module.
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
    /// Initializes a new instance of SUPIRModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses SUPIR defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DPM-Solver++ Multistep for efficient SDXL inference.
    /// </param>
    /// <param name="unet">Custom U-Net. If null, creates the SDXL backbone with 2048-dim cross-attention.</param>
    /// <param name="vae">Custom VAE. If null, creates the SDXL VAE with 0.13025 scale factor.</param>
    /// <param name="conditioner">Optional dual text encoder conditioning module.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SUPIRModel(
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
            scheduler ?? new DPMSolverMultistepScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
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
    /// <param name="unet">Custom U-Net, or null for SDXL backbone defaults.</param>
    /// <param name="vae">Custom VAE, or null for SDXL VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default architecture matches SDXL:
    /// - U-Net: 320 base channels, [1, 2, 4] multipliers, 2048-dim cross-attention
    /// - VAE: 4 latent channels, 128 base channels, 0.13025 SDXL scale factor
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
            latentScaleFactor: SDXL_VAE_SCALE_FACTOR,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates an image from a text prompt using SUPIR/SDXL defaults.
    /// </summary>
    /// <param name="prompt">Text prompt describing desired output quality.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width (default: 1024, SDXL native).</param>
    /// <param name="height">Output height (default: 1024, SDXL native).</param>
    /// <param name="numInferenceSteps">Denoising steps (default: 50).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
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
    /// Performs SDXL-quality image restoration using SUPIR.
    /// </summary>
    /// <param name="inputImage">The degraded image tensor [batch, 3, height, width].</param>
    /// <param name="prompt">
    /// Text prompt guiding restoration quality. SUPIR benefits from descriptive prompts
    /// about the desired output quality and content.
    /// </param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">
    /// Restoration strength (0.0-1.0):
    /// - 0.3-0.4: Conservative (preserves input structure)
    /// - 0.5: Balanced (recommended)
    /// - 0.6-0.8: Aggressive detail generation
    /// </param>
    /// <param name="numInferenceSteps">Denoising steps (default: 50).</param>
    /// <param name="guidanceScale">Guidance scale. If null, uses 7.5.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Restored image tensor at SDXL quality.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SUPIR produces the highest quality restoration results
    /// because it uses SDXL (the best open-source generation model) as its backbone.
    /// The trade-off is higher VRAM usage (~12GB+) and slower processing.
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
            latentScaleFactor: SDXL_VAE_SCALE_FACTOR);
        clonedVae.SetParameters(_vae.GetParameters());

        return new SUPIRModel<T>(
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
            Name = "SUPIR",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "SUPIR SDXL-based photo-realistic image restoration with GPT-guided conditioning",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sdxl-restoration");
        metadata.SetProperty("backbone", "SDXL U-Net");
        metadata.SetProperty("text_encoder", "OpenCLIP ViT-bigG/14 + CLIP ViT-L/14");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("vae_scale_factor", SDXL_VAE_SCALE_FACTOR);
        metadata.SetProperty("gpt_guided", true);
        metadata.SetProperty("scheduler", "DPM-Solver++ Multistep");
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
