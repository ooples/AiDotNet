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
/// Stable Diffusion 3 / SD 3.5 model for text-to-image generation using rectified flow and MMDiT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Diffusion 3 (SD3) is a next-generation text-to-image model by Stability AI that
/// replaces the U-Net with a Multi-Modal Diffusion Transformer (MMDiT) and uses rectified
/// flow matching instead of traditional DDPM-style noise schedules.
/// </para>
/// <para>
/// <b>For Beginners:</b> SD3 is the successor to SDXL with major architectural improvements:
///
/// How SD3 works:
/// 1. Text is encoded by THREE encoders: CLIP ViT-L/14, OpenCLIP ViT-bigG/14, and T5-XXL
/// 2. An MMDiT (Multi-Modal Diffusion Transformer) processes image and text tokens jointly
/// 3. Uses rectified flow instead of DDPM noise scheduling
/// 4. A new 16-channel VAE decodes latents to 1024x1024 images
///
/// Key characteristics:
/// - Triple text encoders (CLIP L + OpenCLIP G + T5-XXL)
/// - MMDiT: Joint attention between text and image tokens
/// - 16 latent channels (vs 4 in SD 1.5/SDXL)
/// - Rectified flow (linear noise schedule, v-prediction)
/// - SD3 Medium: 2B MMDiT parameters, 24 layers
/// - SD3.5 Large: 8B MMDiT parameters, 38 layers
/// - SD3.5 Large Turbo: 8B distilled for 4-step generation
///
/// Advantages:
/// - Superior text rendering in generated images
/// - Better prompt adherence than SDXL
/// - Higher quality details and compositions
/// - Scalable MMDiT architecture
///
/// Limitations:
/// - Higher compute requirements than SDXL
/// - Fewer community fine-tunes (newer ecosystem)
/// - T5-XXL encoder increases VRAM usage significantly
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: MMDiT (Multi-Modal Diffusion Transformer) + 16-channel VAE
/// - Text encoder 1: CLIP ViT-L/14 (768-dim)
/// - Text encoder 2: OpenCLIP ViT-bigG/14 (1280-dim)
/// - Text encoder 3: T5-XXL (4096-dim)
/// - Combined pooled embedding: 2048-dim (768 + 1280)
/// - Context dimension: 4096 (T5 embeddings for cross-attention)
/// - SD3 Medium: 2B params, hidden 1536, 24 layers, 24 heads
/// - SD3.5 Large: 8B params, hidden 2432, 38 layers, 38 heads
/// - VAE: 16 latent channels, scale factor 1.5305, shift 0.0609
/// - Training: Rectified flow matching with linear schedule
/// - Resolution: 1024x1024 (native)
///
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create SD3 Medium with defaults
/// var sd3 = new StableDiffusion3Model&lt;float&gt;();
///
/// // Create SD3.5 Large
/// var sd35 = new StableDiffusion3Model&lt;float&gt;(variant: "3.5-Large");
///
/// // Generate an image
/// var image = sd3.GenerateFromText(
///     prompt: "A serene mountain landscape with the text 'Hello World' carved in stone",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 28,
///     guidanceScale: 7.0,
///     seed: 42);
/// </code>
/// </example>
public class StableDiffusion3Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SD3 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for SD3 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int SD3_LATENT_CHANNELS = 16;
    private const int SD3_CONTEXT_DIM = 4096;
    private const double SD3_DEFAULT_GUIDANCE_SCALE = 7.0;

    #endregion

    #region Fields

    private MMDiTNoisePredictor<T> _mmdit;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly string _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _mmdit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SD3_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _mmdit.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant ("3-Medium", "3.5-Large", "3.5-Large-Turbo").
    /// </summary>
    public string Variant => _variant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableDiffusion3Model with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses SD3 rectified flow defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, creates a flow-matching scheduler.</param>
    /// <param name="mmdit">Custom MMDiT noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Triple text encoder conditioning module.</param>
    /// <param name="variant">Model variant: "3-Medium", "3.5-Large", or "3.5-Large-Turbo" (default: "3-Medium").</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StableDiffusion3Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTNoisePredictor<T>? mmdit = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        string variant = "3-Medium",
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 1.0,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _variant = variant;

        InitializeLayers(mmdit, vae, seed);

        SetGuidanceScale(SD3_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_mmdit), nameof(_vae))]
    private void InitializeLayers(
        MMDiTNoisePredictor<T>? mmdit,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Select architecture based on variant
        var (hiddenSize, numLayers, numHeads) = _variant switch
        {
            "3.5-Large" or "3.5-Large-Turbo" => (2432, 38, 38),
            _ => (1536, 24, 24)  // SD3 Medium
        };

        // MMDiT with joint text-image attention
        _mmdit = mmdit ?? new MMDiTNoisePredictor<T>(
            inputChannels: SD3_LATENT_CHANNELS,
            hiddenSize: hiddenSize,
            numJointLayers: numLayers,
            numHeads: numHeads,
            patchSize: 2,
            contextDim: SD3_CONTEXT_DIM,
            seed: seed);

        // 16-channel VAE with different scale factor than SD 1.5/SDXL
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305,
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
        int numInferenceSteps = 28,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SD3_DEFAULT_GUIDANCE_SCALE;

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
        double strength = 0.75,
        int numInferenceSteps = 28,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SD3_DEFAULT_GUIDANCE_SCALE;

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
        var mmditParams = _mmdit.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = mmditParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < mmditParams.Length; i++)
            combined[i] = mmditParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[mmditParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var mmditCount = _mmdit.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != mmditCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {mmditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var mmditParams = new Vector<T>(mmditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < mmditCount; i++)
            mmditParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[mmditCount + i];

        _mmdit.SetParameters(mmditParams);
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
        var (hiddenSize, numLayers, numHeads) = _variant switch
        {
            "3.5-Large" or "3.5-Large-Turbo" => (2432, 38, 38),
            _ => (1536, 24, 24)
        };

        var clonedMmdit = new MMDiTNoisePredictor<T>(
            inputChannels: SD3_LATENT_CHANNELS,
            hiddenSize: hiddenSize,
            numJointLayers: numLayers,
            numHeads: numHeads,
            patchSize: 2,
            contextDim: SD3_CONTEXT_DIM);
        clonedMmdit.SetParameters(_mmdit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305);
        clonedVae.SetParameters(_vae.GetParameters());

        return new StableDiffusion3Model<T>(
            mmdit: clonedMmdit,
            vae: clonedVae,
            conditioner: _conditioner,
            variant: _variant);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"Stable Diffusion {_variant}",
            Version = _variant,
            ModelType = ModelType.NeuralNetwork,
            Description = $"Stable Diffusion {_variant} with MMDiT architecture, rectified flow, and triple text encoders",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "mmdit-rectified-flow");
        metadata.SetProperty("base_model", "Stable Diffusion 3");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "OpenCLIP ViT-bigG/14");
        metadata.SetProperty("text_encoder_3", "T5-XXL");
        metadata.SetProperty("context_dim", SD3_CONTEXT_DIM);
        metadata.SetProperty("latent_channels", SD3_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("training_framework", "rectified-flow");

        return metadata;
    }

    #endregion
}
