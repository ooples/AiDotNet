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
/// FLUX.2 model for next-generation text-to-image generation by Black Forest Labs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLUX.2 is the successor to FLUX.1, featuring improved image quality, better text rendering,
/// enhanced prompt adherence, and faster inference. It maintains the hybrid MMDiT architecture
/// with improvements to the attention mechanism, flow matching schedule, and training procedure.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLUX.2 is the improved version of FLUX.1, generating even better
/// images with sharper details and more accurate text rendering.
///
/// How FLUX.2 works:
/// 1. Text is encoded by CLIP ViT-L/14 and T5-XXL encoders (dual encoder design)
/// 2. A hybrid MMDiT with 19 joint blocks + 38 single blocks processes tokens
/// 3. Improved rectified flow matching enables efficient 28-step generation
/// 4. A 16-channel VAE decodes latents to high-resolution images
///
/// Model variants:
/// - FLUX.2 [pro]: Best quality, API-only, not open-source
/// - FLUX.2 [dev]: Open-weight, guidance-distilled, non-commercial license
/// - FLUX.2 [schnell]: Fast 1-4 step generation, Apache 2.0 license
///
/// Key improvements over FLUX.1:
/// - Better text rendering and prompt adherence
/// - Higher image quality at fewer inference steps (28 vs 50)
/// - Improved color accuracy and composition
/// - Enhanced fine detail generation
///
/// Technical characteristics:
/// - 12B parameters in the transformer
/// - Hybrid architecture: 19 double-stream + 38 single-stream blocks
/// - Hidden size: 3072, 24 attention heads
/// - Dual text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
/// - 16 latent channels with improved VAE
/// - Rotary Position Embeddings (RoPE)
/// - Improved rectified flow with optimized noise schedule
///
/// Limitations:
/// - Very high VRAM requirements (~24GB for dev)
/// - pro variant is API-only
/// - Newer model with evolving community support
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.2 Technical Report", 2025
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create FLUX.2 dev (open-weight variant)
/// var flux2 = new Flux2Model&lt;float&gt;();
///
/// // Create FLUX.2 schnell for fast generation
/// var flux2Schnell = new Flux2Model&lt;float&gt;(variant: FluxVariant.Schnell);
///
/// // Generate an image
/// var image = flux2.GenerateFromText(
///     prompt: "A photorealistic landscape with aurora borealis over snow-capped mountains",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 28,
///     guidanceScale: 3.5,
///     seed: 42);
/// </code>
/// </example>
public class Flux2Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for FLUX.2 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for FLUX.2 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int FLUX2_LATENT_CHANNELS = 16;
    private const int FLUX2_HIDDEN_SIZE = 3072;
    private const int FLUX2_JOINT_LAYERS = 19;
    private const int FLUX2_SINGLE_LAYERS = 38;
    private const int FLUX2_NUM_HEADS = 24;
    private const int FLUX2_CONTEXT_DIM = 4096;
    private const double FLUX2_DEV_GUIDANCE_SCALE = 3.5;
    private const double FLUX2_SCHNELL_GUIDANCE_SCALE = 0.0;
    private const int FLUX2_DEV_DEFAULT_STEPS = 28;
    private const int FLUX2_SCHNELL_DEFAULT_STEPS = 4;

    #endregion

    #region Fields

    private FluxDoubleStreamPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly FluxVariant _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => FLUX2_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant (Dev, Schnell, or Pro).
    /// </summary>
    public FluxVariant Variant => _variant;

    /// <summary>
    /// Gets whether this variant supports guidance-free generation.
    /// </summary>
    public bool IsGuidanceFree => _variant == FluxVariant.Schnell;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Flux2Model with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses FLUX.2 rectified flow defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom FLUX double-stream noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Dual text encoder conditioning module (CLIP + T5).</param>
    /// <param name="variant">Model variant: Dev (default), Schnell, or Pro.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Flux2Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FluxDoubleStreamPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        FluxVariant variant = FluxVariant.Dev,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 1.0,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        _variant = variant;

        InitializeLayers(predictor, vae, seed);

        var guidanceScale = variant == FluxVariant.Schnell
            ? FLUX2_SCHNELL_GUIDANCE_SCALE
            : FLUX2_DEV_GUIDANCE_SCALE;
        SetGuidanceScale(guidanceScale);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        FluxDoubleStreamPredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        // FLUX.2 uses improved double-stream predictor (V2 variant)
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(
            variant: FluxPredictorVariant.V2,
            seed: seed);

        // 16-channel VAE (same family as FLUX.1/SD3)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: FLUX2_LATENT_CHANNELS,
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
        int numInferenceSteps = FLUX2_DEV_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Schnell variant uses 4 steps by default with no guidance
        var effectiveSteps = _variant == FluxVariant.Schnell && numInferenceSteps == FLUX2_DEV_DEFAULT_STEPS
            ? FLUX2_SCHNELL_DEFAULT_STEPS
            : numInferenceSteps;

        var effectiveGuidanceScale = guidanceScale ??
            (_variant == FluxVariant.Schnell ? FLUX2_SCHNELL_GUIDANCE_SCALE : FLUX2_DEV_GUIDANCE_SCALE);

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
        int numInferenceSteps = FLUX2_DEV_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ??
            (_variant == FluxVariant.Schnell ? FLUX2_SCHNELL_GUIDANCE_SCALE : FLUX2_DEV_GUIDANCE_SCALE);

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
        var clonedPredictor = new FluxDoubleStreamPredictor<T>(
            variant: FluxPredictorVariant.V2);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: FLUX2_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Flux2Model<T>(
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
        var metadata = new ModelMetadata<T>
        {
            Name = $"FLUX.2 [{_variant}]",
            Version = _variant.ToString(),
            ModelType = ModelType.NeuralNetwork,
            Description = $"FLUX.2 [{_variant}] next-generation hybrid MMDiT with {FLUX2_JOINT_LAYERS} joint + {FLUX2_SINGLE_LAYERS} single blocks and improved rectified flow",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "hybrid-mmdit-rectified-flow-v2");
        metadata.SetProperty("base_model", "FLUX.2");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "T5-XXL");
        metadata.SetProperty("context_dim", FLUX2_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", FLUX2_HIDDEN_SIZE);
        metadata.SetProperty("joint_layers", FLUX2_JOINT_LAYERS);
        metadata.SetProperty("single_layers", FLUX2_SINGLE_LAYERS);
        metadata.SetProperty("num_heads", FLUX2_NUM_HEADS);
        metadata.SetProperty("latent_channels", FLUX2_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("guidance_free", _variant == FluxVariant.Schnell);
        metadata.SetProperty("default_guidance_scale", _variant == FluxVariant.Schnell ? FLUX2_SCHNELL_GUIDANCE_SCALE : FLUX2_DEV_GUIDANCE_SCALE);
        metadata.SetProperty("default_inference_steps", _variant == FluxVariant.Schnell ? FLUX2_SCHNELL_DEFAULT_STEPS : FLUX2_DEV_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
