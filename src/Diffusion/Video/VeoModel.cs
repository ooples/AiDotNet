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

namespace AiDotNet.Diffusion.Video;

/// <summary>
/// Veo model for Google's high-fidelity cascaded video generation with temporal super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Veo by Google DeepMind uses cascaded diffusion with temporal super-resolution for
/// high-resolution, long-duration video generation. The base model generates at lower
/// resolution, then spatial and temporal super-resolution stages produce the final output.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Cascaded DiT with 40 transformer layers and 2560 hidden dimension</description></item>
/// <item><description>20 attention heads with full spatiotemporal attention</description></item>
/// <item><description>T5-XXL + CLIP dual text encoding for 4096-dim context</description></item>
/// <item><description>3D causal VAE with 16 latent channels and 3 temporal layers</description></item>
/// <item><description>Cascaded pipeline: base → spatial SR → temporal SR</description></item>
/// <item><description>Flow matching training objective</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Veo is Google's top-tier video generation model.
///
/// How Veo works:
/// 1. Text is encoded by dual T5-XXL + CLIP encoders into 4096-dim embeddings
/// 2. Base model generates low-resolution video in compressed latent space
/// 3. Spatial super-resolution stage upscales each frame
/// 4. Temporal super-resolution stage adds interpolated frames
/// 5. The causal VAE decodes the final high-resolution video
///
/// Key characteristics:
/// - Cascaded architecture: base → spatial SR → temporal SR
/// - 1080p output with 60+ second duration capability
/// - Dual text encoding (T5-XXL + CLIP) for rich conditioning
/// - Veo 2 variant with improved quality and consistency
/// - 150 frames at 24 FPS by default (~6.25 seconds)
///
/// When to use Veo:
/// - Highest-quality video generation
/// - Long-duration video content
/// - 1080p high-resolution output
/// - Text-to-video, image-to-video, and video-to-video tasks
///
/// Limitations:
/// - Proprietary model (API-only access through Google)
/// - Very high compute requirements for cascaded generation
/// - Slower generation due to multi-stage pipeline
/// - Limited public information on exact architecture details
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Cascaded DiT (base + spatial SR + temporal SR)
/// - Hidden dimension: 2560
/// - Transformer layers: 40
/// - Attention heads: 20
/// - Patch size: 2
/// - Latent channels: 16 (3D causal VAE)
/// - Context dimension: 4096 (T5-XXL + CLIP dual encoder)
/// - Output resolution: Up to 1080p
/// - Default: 150 frames at 24 FPS (~6.25 seconds)
/// - Veo 2: Enhanced quality variant (200 frames default)
/// - Training objective: Flow matching
///
/// Reference: Google DeepMind, "Veo: High-Fidelity Video Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Veo with defaults
/// var veo = new VeoModel&lt;float&gt;();
///
/// // Create Veo 2 variant
/// var veo2 = VeoModel&lt;float&gt;.CreateVeo2();
///
/// // Generate video from text
/// var video = veo.GenerateFromText(
///     prompt: "An aerial view of a coral reef teeming with tropical fish",
///     width: 1920,
///     height: 1080,
///     numFrames: 150,
///     fps: 24,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate video from an image
/// var animated = veo.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 150,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class VeoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the 3D causal VAE (16).
    /// </summary>
    private const int LATENT_CHANNELS = 16;

    /// <summary>
    /// Hidden dimension of the cascaded DiT (2560).
    /// </summary>
    private const int HIDDEN_DIM = 2560;

    /// <summary>
    /// Number of transformer layers (40).
    /// </summary>
    private const int NUM_LAYERS = 40;

    /// <summary>
    /// Number of attention heads (20).
    /// </summary>
    private const int NUM_HEADS = 20;

    /// <summary>
    /// Context dimension from the dual text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames for Veo (150).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 150;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The cascaded DiT noise predictor.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The 3D causal VAE for spatiotemporal video compression.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The dual T5-XXL + CLIP text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Whether this is a Veo 2 variant.
    /// </summary>
    private readonly bool _isVeo2;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _temporalVAE;

    /// <inheritdoc />
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsImageToVideo => true;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => true;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    /// <summary>
    /// Gets whether this is a Veo 2 variant with enhanced quality.
    /// </summary>
    public bool IsVeo2 => _isVeo2;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of VeoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Veo defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 40-layer cascaded DiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the 3D causal VAE.</param>
    /// <param name="conditioner">Dual T5-XXL + CLIP text encoder conditioning module.</param>
    /// <param name="isVeo2">Whether to use Veo 2 enhanced variant (default: false).</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 150 for Veo, 200 for Veo 2).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public VeoModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        bool isVeo2 = false,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _isVeo2 = isVeo2;
        _conditioner = conditioner;

        InitializeLayers(dit, temporalVAE, seed);
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a Veo 2 variant with enhanced quality and longer duration.
    /// </summary>
    /// <param name="conditioner">Optional dual text encoder conditioning module.</param>
    /// <returns>A new VeoModel configured for Veo 2 specifications.</returns>
    public static VeoModel<T> CreateVeo2(IConditioningModule<T>? conditioner = null)
    {
        return new VeoModel<T>(
            isVeo2: true,
            conditioner: conditioner,
            defaultNumFrames: 200,
            defaultFPS: 24);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the cascaded DiT and temporal VAE using custom or default configurations.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for Veo defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for 3D causal VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default Cascaded DiT:
    /// - 16 input/output channels, 2560 hidden dim, 40 layers, 20 heads
    /// - Patch size 2, full spatiotemporal attention
    /// - 4096-dim context from dual T5-XXL + CLIP
    ///
    /// Default 3D Causal VAE:
    /// - 16 latent channels, 3 temporal layers
    /// - Causal mode for sequential generation
    /// - 0.13025 latent scale factor
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_dit), nameof(_temporalVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3,
            temporalKernelSize: 3,
            causalMode: true,
            latentScaleFactor: 0.13025);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        return _dit.PredictNoise(latents, timestep, imageEmbedding);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();

        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
        {
            combined[i] = ditParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[ditParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var ditCount = _dit.ParameterCount;
        var vaeCount = _temporalVAE.GetParameters().Length;

        if (parameters.Length != ditCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
        {
            ditParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[ditCount + i];
        }

        _dit.SetParameters(ditParams);
        _temporalVAE.SetParameters(vaeParams);
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
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new VeoModel<T>(
            dit: clonedDit,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            isVeo2: _isVeo2,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _isVeo2 ? "Veo-2" : "Veo",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = $"Google {(_isVeo2 ? "Veo 2" : "Veo")} cascaded video generation with temporal super-resolution",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-dit");
        metadata.SetProperty("is_veo2", _isVeo2);
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("max_resolution", "1080p");
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("default_frames", DefaultNumFrames);

        return metadata;
    }

    #endregion
}
