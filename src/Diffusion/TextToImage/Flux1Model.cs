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
/// FLUX.1 model for high-quality text-to-image generation by Black Forest Labs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLUX.1 is a state-of-the-art text-to-image model developed by Black Forest Labs
/// (founded by Stability AI alumni). It uses a hybrid MMDiT architecture with both
/// double-stream (joint attention) and single-stream transformer blocks, plus rectified
/// flow matching for training.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLUX.1 represents the cutting edge of open text-to-image models:
///
/// How FLUX.1 works:
/// 1. Text is encoded by CLIP ViT-L/14 and T5-XXL encoders
/// 2. A hybrid MMDiT with 19 joint blocks + 38 single blocks processes tokens
/// 3. Rectified flow matching enables efficient generation
/// 4. A 16-channel VAE decodes latents to high-resolution images
///
/// Model variants:
/// - FLUX.1 [pro]: Best quality, API-only, not open-source
/// - FLUX.1 [dev]: Open-weight, guidance-distilled, non-commercial license
/// - FLUX.1 [schnell]: Fast 1-4 step generation, Apache 2.0 license
///
/// Key characteristics:
/// - 12B parameters in the transformer
/// - Hybrid architecture: 19 double-stream + 38 single-stream blocks
/// - Hidden size: 3072, 24 attention heads
/// - Dual text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
/// - 16 latent channels with new VAE
/// - Rotary Position Embeddings (RoPE)
/// - Rectified flow with linear noise schedule
///
/// Advantages:
/// - State-of-the-art image quality
/// - Excellent text rendering
/// - Superior prompt adherence
/// - schnell variant: very fast (1-4 steps)
/// - dev variant: open weights for research
///
/// Limitations:
/// - Very high VRAM requirements (~24GB for dev)
/// - pro variant is API-only
/// - Newer ecosystem (fewer community tools)
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Hybrid MMDiT (double-stream + single-stream blocks)
/// - Transformer: 12B params, hidden 3072, 19 joint + 38 single layers, 24 heads
/// - Text encoder 1: CLIP ViT-L/14 (768-dim, pooled embeddings)
/// - Text encoder 2: T5-XXL (4096-dim, sequence embeddings)
/// - Context dimension: 4096 (T5 embeddings)
/// - Patch size: 2 (in latent space)
/// - VAE: 16 latent channels
/// - Training: Rectified flow matching
/// - dev: 50-step guidance-distilled
/// - schnell: 1-4 step distilled
/// - Resolution: Up to 2048x2048 (aspect-ratio aware)
///
/// Reference: Black Forest Labs, "FLUX.1 Technical Report", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create FLUX.1 dev (open-weight variant)
/// var flux = new Flux1Model&lt;float&gt;();
///
/// // Create FLUX.1 schnell for fast generation
/// var fluxSchnell = new Flux1Model&lt;float&gt;(variant: "schnell");
///
/// // Generate an image
/// var image = flux.GenerateFromText(
///     prompt: "A photorealistic portrait of a woman with flowing red hair in golden light",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 3.5,
///     seed: 42);
/// </code>
/// </example>
public class Flux1Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for FLUX.1 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for FLUX.1 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int FLUX_LATENT_CHANNELS = 16;
    private const int FLUX_HIDDEN_SIZE = 3072;
    private const int FLUX_JOINT_LAYERS = 19;
    private const int FLUX_SINGLE_LAYERS = 38;
    private const int FLUX_NUM_HEADS = 24;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double FLUX_DEV_GUIDANCE_SCALE = 3.5;
    private const double FLUX_SCHNELL_GUIDANCE_SCALE = 0.0;

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
    public override int LatentChannels => FLUX_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _mmdit.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant ("dev", "schnell", or "pro").
    /// </summary>
    public string Variant => _variant;

    /// <summary>
    /// Gets whether this variant supports guidance-free generation.
    /// </summary>
    public bool IsGuidanceFree => _variant == "schnell";

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Flux1Model with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses FLUX rectified flow defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="mmdit">Custom MMDiT noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Dual text encoder conditioning module (CLIP + T5).</param>
    /// <param name="variant">Model variant: "dev", "schnell", or "pro" (default: "dev").</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Flux1Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTNoisePredictor<T>? mmdit = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        string variant = "dev",
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

        var guidanceScale = variant == "schnell" ? FLUX_SCHNELL_GUIDANCE_SCALE : FLUX_DEV_GUIDANCE_SCALE;
        SetGuidanceScale(guidanceScale);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_mmdit), nameof(_vae))]
    private void InitializeLayers(
        MMDiTNoisePredictor<T>? mmdit,
        StandardVAE<T>? vae,
        int? seed)
    {
        // FLUX.1 uses hybrid MMDiT with both joint and single-stream blocks
        _mmdit = mmdit ?? new MMDiTNoisePredictor<T>(
            inputChannels: FLUX_LATENT_CHANNELS,
            hiddenSize: FLUX_HIDDEN_SIZE,
            numJointLayers: FLUX_JOINT_LAYERS,
            numSingleLayers: FLUX_SINGLE_LAYERS,
            numHeads: FLUX_NUM_HEADS,
            patchSize: 2,
            contextDim: FLUX_CONTEXT_DIM,
            seed: seed);

        // 16-channel VAE (same architecture as SD3)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: FLUX_LATENT_CHANNELS,
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
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        // schnell variant uses 4 steps by default, no guidance
        var effectiveSteps = _variant == "schnell" && numInferenceSteps == 50
            ? 4
            : numInferenceSteps;

        var effectiveGuidanceScale = guidanceScale ??
            (_variant == "schnell" ? FLUX_SCHNELL_GUIDANCE_SCALE : FLUX_DEV_GUIDANCE_SCALE);

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
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ??
            (_variant == "schnell" ? FLUX_SCHNELL_GUIDANCE_SCALE : FLUX_DEV_GUIDANCE_SCALE);

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
        var clonedMmdit = new MMDiTNoisePredictor<T>(
            inputChannels: FLUX_LATENT_CHANNELS,
            hiddenSize: FLUX_HIDDEN_SIZE,
            numJointLayers: FLUX_JOINT_LAYERS,
            numSingleLayers: FLUX_SINGLE_LAYERS,
            numHeads: FLUX_NUM_HEADS,
            patchSize: 2,
            contextDim: FLUX_CONTEXT_DIM);
        clonedMmdit.SetParameters(_mmdit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: FLUX_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Flux1Model<T>(
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
            Name = $"FLUX.1 [{_variant}]",
            Version = _variant,
            ModelType = ModelType.NeuralNetwork,
            Description = $"FLUX.1 [{_variant}] hybrid MMDiT with {FLUX_JOINT_LAYERS} joint + {FLUX_SINGLE_LAYERS} single blocks and rectified flow",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "hybrid-mmdit-rectified-flow");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "T5-XXL");
        metadata.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", FLUX_HIDDEN_SIZE);
        metadata.SetProperty("joint_layers", FLUX_JOINT_LAYERS);
        metadata.SetProperty("single_layers", FLUX_SINGLE_LAYERS);
        metadata.SetProperty("latent_channels", FLUX_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("guidance_free", _variant == "schnell");

        return metadata;
    }

    #endregion
}
