using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// OmniGen model — unified image generation model handling multiple tasks in one architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OmniGen is a unified image generation model that handles text-to-image, image editing,
/// subject-driven generation, and visual conditional generation with a single model,
/// without requiring task-specific adapters or fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> OmniGen is one model that does everything:
///
/// Key characteristics:
/// - Unified model: text-to-image, editing, inpainting, subject-driven, etc.
/// - No task-specific adapters needed (unlike ControlNet, IP-Adapter)
/// - Single transformer backbone handles all tasks
/// - Interleaved image-text input for flexible conditioning
/// - In-context learning: understands task from examples
///
/// Tasks OmniGen can handle:
/// - Text-to-image generation
/// - Image editing (instruction-based)
/// - Subject-driven generation (given reference images)
/// - Visual conditional generation (depth, edge, pose)
/// - Style transfer
///
/// Use OmniGen when you need:
/// - Single model for multiple generation tasks
/// - Simplified pipeline (no adapter management)
/// - Flexible conditioning from images and text
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Unified transformer with interleaved image-text tokens
/// - Parameters: ~3.8B
/// - Text encoder: integrated (not separate)
/// - Resolution: 512x512 to 1024x1024
/// - Latent channels: 4, 8x downsampling
///
/// Reference: Xiao et al., "OmniGen: Unified Image Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var omniGen = new OmniGenModel&lt;float&gt;();
///
/// // Create with custom architecture for pre-trained weight loading
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;();
/// var customOmniGen = new OmniGenModel&lt;float&gt;(architecture: architecture);
///
/// // Generate a 1024x1024 image from text
/// var image = omniGen.GenerateFromText(
///     prompt: "A futuristic cityscape at dusk with neon lights reflecting off wet streets",
///     negativePrompt: "blurry, low quality",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 3.0,
///     seed: 42);
///
/// // Image-to-image transformation
/// var transformed = omniGen.ImageToImage(
///     inputImage: existingImage,
///     prompt: "Convert to watercolor painting style",
///     strength: 0.7);
/// </code>
/// </example>
public class OmniGenModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for OmniGen generation (1024 pixels).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for OmniGen generation (1024 pixels).
    /// </summary>
    public const int DefaultHeight = 1024;

    /// <summary>
    /// Number of latent channels in OmniGen's VAE (4 channels).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension matching the unified transformer's context size (2048).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 2048;

    /// <summary>
    /// Default guidance scale for OmniGen (3.0, lower than typical due to unified architecture).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 3.0;

    #endregion

    #region Fields

    /// <summary>
    /// The DiT noise predictor (unified transformer backbone, ~3.8B parameters).
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The VAE for encoding images to latent space and decoding back.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// Optional conditioning module for text/image conditioning.
    /// </summary>
    /// <remarks>
    /// OmniGen integrates its text encoder, so this is optional and used
    /// for external conditioning modules if needed.
    /// </remarks>
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of OmniGenModel with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture for pre-trained weight loading.
    /// If null, creates architecture from default OmniGen specifications.
    /// </param>
    /// <param name="options">
    /// Configuration options for the diffusion model. If null, uses OmniGen defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDPM scheduler with default settings.
    /// </param>
    /// <param name="dit">
    /// Custom DiT noise predictor. If null, creates the standard OmniGen DiT
    /// with 2048 hidden size, 32 layers, 16 heads, patch size 2, and 2048-dim cross-attention.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard VAE with 128 base channels,
    /// [1, 2, 4, 4] multipliers, 4 latent channels, and 0.18215 scale factor.
    /// </param>
    /// <param name="conditioner">
    /// Optional conditioning module for external text/image conditioning.
    /// </param>
    /// <param name="seed">
    /// Optional random seed for reproducible generation.
    /// </param>
    public OmniGenModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
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
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(dit, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the DiT and VAE layers, using custom components if provided
    /// or creating industry-standard layers from the OmniGen specifications.
    /// </summary>
    /// <param name="dit">Custom DiT noise predictor, or null to create the standard OmniGen DiT.</param>
    /// <param name="vae">Custom VAE, or null to create the standard OmniGen VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(DiTNoisePredictor<T>? dit, StandardVAE<T>? vae, int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2048,
            numLayers: 32,
            numHeads: 16,
            patchSize: 2,
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
        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            guidanceScale ?? DEFAULT_GUIDANCE_SCALE,
            seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
            combined[i] = ditParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[ditParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int ditCount = _dit.ParameterCount;
        int vaeCount = _vae.ParameterCount;

        if (parameters.Length != ditCount + vaeCount)
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
            ditParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[ditCount + i];

        _dit.SetParameters(ditParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2048,
            numLayers: 32,
            numHeads: 16,
            patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new OmniGenModel<T>(
            dit: clonedDit,
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
            Name = "OmniGen",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "OmniGen unified multi-task image generation model",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unified-dit");
        metadata.SetProperty("tasks", "text2img,editing,subject-driven,conditional");
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("hidden_size", 2048);
        metadata.SetProperty("num_layers", 32);
        metadata.SetProperty("num_heads", 16);
        metadata.SetProperty("patch_size", 2);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE_SCALE);

        return metadata;
    }

    #endregion
}
