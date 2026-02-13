using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// Latent Consistency Model (LCM) for fast few-step image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Latent Consistency Models (LCM) distill pre-trained latent diffusion models into
/// fast inference models that can generate high-quality images in 2-8 steps.
/// LCM uses consistency distillation in latent space to learn a one-step mapping.
/// </para>
/// <para>
/// <b>For Beginners:</b> LCM makes Stable Diffusion much faster:
///
/// How LCM works:
/// 1. Starts with a pre-trained SD model (1.5, 2.1, or SDXL)
/// 2. Uses consistency distillation to learn a shortcut through the denoising process
/// 3. The distilled model can skip most denoising steps (2-8 instead of 20-50)
/// 4. LCM-LoRA allows applying this speedup to ANY fine-tuned model
///
/// Key characteristics:
/// - 2-8 step generation (vs 20-50 for standard SD)
/// - Based on consistency distillation (not adversarial training)
/// - Lower guidance scales needed (1.0-2.0 vs 7.5)
/// - LCM-LoRA variant: lightweight adapter compatible with any SD fine-tune
/// - Compatible with SD 1.5, SD 2.1, and SDXL base models
///
/// Advantages:
/// - 5-10x faster than standard SD
/// - LCM-LoRA is composable with other LoRAs
/// - Maintains good quality at 4 steps
/// - Open-source and widely available
///
/// Limitations:
/// - Slightly lower diversity than full-step models
/// - Very low step counts (1-2) may show artifacts
/// - Lower guidance scales may reduce prompt adherence
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Consistency-distilled latent diffusion model
/// - Base: SD 1.5/2.1/SDXL U-Net with consistency training
/// - Distillation: Latent Consistency Distillation (LCD)
/// - Optimal steps: 4 (good quality/speed tradeoff)
/// - Guidance scale: 1.0-2.0 (lower than standard SD)
/// - Scheduler: LCM scheduler with skipping timesteps
/// - LCM-LoRA: ~67M adapter parameters for any SD fine-tune
///
/// Reference: Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create LCM for fast generation
/// var lcm = new LatentConsistencyModel&lt;float&gt;();
///
/// // Generate in just 4 steps
/// var image = lcm.GenerateFromText(
///     prompt: "A beautiful landscape painting",
///     width: 512,
///     height: 512,
///     numInferenceSteps: 4,
///     guidanceScale: 1.5,
///     seed: 42);
/// </code>
/// </example>
public class LatentConsistencyModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for LCM (matches SD 1.5).
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for LCM (matches SD 1.5).
    /// </summary>
    public const int DefaultHeight = 512;

    private const int LCM_LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension (768 for SD 1.5 base, 1024 for SD 2.1 base).
    /// </summary>
    private const int LCM_CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default guidance scale for LCM (1.5, much lower than standard SD).
    /// </summary>
    private const double LCM_DEFAULT_GUIDANCE_SCALE = 1.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly string _baseModel;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LCM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the base model identifier ("SD1.5", "SD2.1", or "SDXL").
    /// </summary>
    public string BaseModel => _baseModel;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of LatentConsistencyModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses LCM defaults.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler configured for LCM.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net. If null, creates a consistency-distilled SD 1.5 U-Net.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard SD VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module.
    /// </param>
    /// <param name="baseModel">
    /// Base model identifier: "SD1.5", "SD2.1", or "SDXL" (default: "SD1.5").
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public LatentConsistencyModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        string baseModel = "SD1.5",
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        _baseModel = baseModel;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(LCM_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the consistency-distilled U-Net and VAE layers,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the LCM paper.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        var contextDim = _baseModel switch
        {
            "SDXL" => 2048,
            "SD2.1" => 1024,
            _ => LCM_CROSS_ATTENTION_DIM
        };

        // Consistency-distilled U-Net (same architecture as base, distilled weights)
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LCM_LATENT_CHANNELS,
            outputChannels: LCM_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: contextDim,
            seed: seed);

        // Standard SD VAE
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LCM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 4,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? LCM_DEFAULT_GUIDANCE_SCALE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 4,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? LCM_DEFAULT_GUIDANCE_SCALE;

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
        var contextDim = _baseModel switch
        {
            "SDXL" => 2048,
            "SD2.1" => 1024,
            _ => LCM_CROSS_ATTENTION_DIM
        };

        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LCM_LATENT_CHANNELS,
            outputChannels: LCM_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: contextDim);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LCM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new LatentConsistencyModel<T>(
            unet: clonedUnet,
            vae: clonedVae,
            conditioner: _conditioner,
            baseModel: _baseModel);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Latent Consistency Model",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = $"Latent Consistency Model distilled from {_baseModel} for fast 2-8 step generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "consistency-distilled-latent-diffusion");
        metadata.SetProperty("base_model", _baseModel);
        metadata.SetProperty("optimal_steps", 4);
        metadata.SetProperty("guidance_scale", LCM_DEFAULT_GUIDANCE_SCALE);
        metadata.SetProperty("latent_channels", LCM_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
