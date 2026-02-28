using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Stable Diffusion 3.5 model with improved MMDiT-X architecture by Stability AI.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SD 3.5 builds on SD3's MMDiT architecture with improved text-image alignment,
/// better detail generation, and QK-normalization for training stability. Uses
/// rectified flow matching training and triple text encoders for comprehensive
/// prompt understanding. Available in Medium (2.5B) and Large (8B) variants.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stable Diffusion 3.5 is Stability AI's latest open model.
///
/// How SD 3.5 works:
/// 1. Text is encoded by three encoders: CLIP ViT-L/14, OpenCLIP ViT-bigG, and T5-XXL
/// 2. An improved MMDiT-X transformer with QK-normalization processes text and image tokens
/// 3. Rectified flow matching enables efficient 28-40 step generation
/// 4. A 16-channel VAE decodes latents to high-resolution images
///
/// Model variants:
/// - SD 3.5 Medium: 2.5B parameters, faster, good quality-speed tradeoff
/// - SD 3.5 Large: 8B parameters, highest quality, more VRAM required
///
/// Key characteristics:
/// - MMDiT-X architecture with QK-normalization for stable training
/// - Triple text encoders for superior prompt understanding
/// - 16 latent channels with improved VAE
/// - Rectified flow matching (not DDPM/DDIM)
/// - Medium: 2.5B params, 1536 hidden, 24 layers
/// - Large: 8B params, 4096 hidden, 38 layers
///
/// Advantages:
/// - Open-weight (Stability AI Community License)
/// - State-of-the-art quality among open models
/// - Better prompt adherence than SDXL
/// - QK-norm prevents training instabilities
///
/// Limitations:
/// - Large variant requires significant VRAM (~24GB)
/// - Newer ecosystem than SD 1.5/SDXL
/// - Commercial use requires separate license agreement
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: MMDiT-X with QK-normalization
/// - Medium: 2.5B params, hidden 1536, 24 layers, 24 heads
/// - Large: 8B params, hidden 4096, 38 layers, 64 heads
/// - Text encoder 1: CLIP ViT-L/14 (768-dim, pooled)
/// - Text encoder 2: OpenCLIP ViT-bigG (1280-dim, pooled)
/// - Text encoder 3: T5-XXL (4096-dim, sequence)
/// - Patch size: 2 (in latent space)
/// - VAE: 16 latent channels, 8x spatial compression
/// - Training: Rectified flow matching
/// - Resolution: 1024x1024 default, up to 2048x2048
///
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for
/// High-Resolution Image Synthesis", ICML 2024 (SD3 base);
/// Stability AI, "Stable Diffusion 3.5 Release Notes", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create SD 3.5 Medium (default)
/// var sd35 = new StableDiffusion35Model&lt;float&gt;();
///
/// // Create SD 3.5 Large for higher quality
/// var sd35Large = new StableDiffusion35Model&lt;float&gt;(variant: MMDiTXVariant.Large);
///
/// // Generate an image
/// var image = sd35.GenerateFromText(
///     prompt: "A detailed oil painting of a medieval castle at dawn",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 28,
///     guidanceScale: 7.0,
///     seed: 42);
/// </code>
/// </example>
public class StableDiffusion35Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SD 3.5 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for SD 3.5 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int SD35_LATENT_CHANNELS = 16;
    private const int SD35_MEDIUM_HIDDEN_SIZE = 1536;
    private const int SD35_LARGE_HIDDEN_SIZE = 4096;
    private const int SD35_MEDIUM_NUM_LAYERS = 24;
    private const int SD35_LARGE_NUM_LAYERS = 38;
    private const int SD35_MEDIUM_NUM_HEADS = 24;
    private const int SD35_LARGE_NUM_HEADS = 64;
    private const int SD35_CONTEXT_DIM = 4096;
    private const double SD35_DEFAULT_GUIDANCE = 7.0;
    private const int SD35_MEDIUM_DEFAULT_STEPS = 28;
    private const int SD35_LARGE_DEFAULT_STEPS = 40;

    #endregion

    #region Fields

    private MMDiTXNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly MMDiTXVariant _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SD35_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant (Medium or Large).
    /// </summary>
    public MMDiTXVariant Variant => _variant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableDiffusion35Model.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses SD 3.5 rectified flow defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom MMDiT-X noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Triple text encoder conditioning module (CLIP + OpenCLIP + T5).</param>
    /// <param name="variant">Model variant: Medium (default, 2.5B) or Large (8B).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StableDiffusion35Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTXNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        MMDiTXVariant variant = MMDiTXVariant.Medium,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        _variant = variant;

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(SD35_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        MMDiTXNoisePredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(
            variant: _variant,
            seed: seed);

        // 16-channel VAE (same as SD3)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD35_LATENT_CHANNELS,
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
        // Use variant-appropriate default step count
        var effectiveSteps = numInferenceSteps == 28 && _variant == MMDiTXVariant.Large
            ? SD35_LARGE_DEFAULT_STEPS
            : numInferenceSteps;

        var effectiveGuidanceScale = guidanceScale ?? SD35_DEFAULT_GUIDANCE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            effectiveSteps,
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
        var effectiveGuidanceScale = guidanceScale ?? SD35_DEFAULT_GUIDANCE;

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
        var predictorParams = _predictor.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = predictorParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < predictorParams.Length; i++)
            combined[i] = predictorParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[predictorParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var predictorCount = _predictor.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != predictorCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {predictorCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var predictorParams = new Vector<T>(predictorCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < predictorCount; i++)
            predictorParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[predictorCount + i];

        _predictor.SetParameters(predictorParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedPredictor = new MMDiTXNoisePredictor<T>(variant: _variant);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD35_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305);
        clonedVae.SetParameters(_vae.GetParameters());

        return new StableDiffusion35Model<T>(
            predictor: clonedPredictor,
            vae: clonedVae,
            conditioner: _conditioner,
            variant: _variant);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        bool isLarge = _variant == MMDiTXVariant.Large;
        var metadata = new ModelMetadata<T>
        {
            Name = $"Stable Diffusion 3.5 [{_variant}]",
            Version = "3.5",
            ModelType = ModelType.NeuralNetwork,
            Description = $"SD 3.5 {_variant} with MMDiT-X ({(isLarge ? "8B" : "2.5B")} params), QK-normalization, and triple text encoders",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "mmdit-x-rectified-flow");
        metadata.SetProperty("base_model", "Stable Diffusion 3.5");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "OpenCLIP ViT-bigG");
        metadata.SetProperty("text_encoder_3", "T5-XXL");
        metadata.SetProperty("context_dim", SD35_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", isLarge ? SD35_LARGE_HIDDEN_SIZE : SD35_MEDIUM_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", isLarge ? SD35_LARGE_NUM_LAYERS : SD35_MEDIUM_NUM_LAYERS);
        metadata.SetProperty("num_heads", isLarge ? SD35_LARGE_NUM_HEADS : SD35_MEDIUM_NUM_HEADS);
        metadata.SetProperty("latent_channels", SD35_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("default_guidance_scale", SD35_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", isLarge ? SD35_LARGE_DEFAULT_STEPS : SD35_MEDIUM_DEFAULT_STEPS);
        metadata.SetProperty("qk_normalization", true);

        return metadata;
    }

    #endregion
}
