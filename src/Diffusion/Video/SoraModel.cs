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
/// Sora-architecture model for DiT-based video generation with native spatiotemporal patches.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sora by OpenAI uses a Diffusion Transformer (DiT) operating on spatiotemporal patches
/// of video, enabling native variable-duration, resolution, and aspect-ratio video generation.
/// The model treats videos as sequences of spacetime patches, similar to how LLMs process tokens.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>DiT backbone with 48 transformer layers and 3072 hidden dimension</description></item>
/// <item><description>24 attention heads with full 3D spatiotemporal attention</description></item>
/// <item><description>3D spatiotemporal patch embeddings (patch size 2)</description></item>
/// <item><description>3D causal VAE with 16 latent channels for spatiotemporal compression</description></item>
/// <item><description>4096-dim text conditioning (CLIP + T5 dual encoder)</description></item>
/// <item><description>Flow matching training objective</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Sora creates videos from text using a world-simulator approach.
///
/// How Sora works:
/// 1. Text is encoded by dual CLIP + T5 encoders into 4096-dim embeddings
/// 2. Video is compressed by the 3D causal VAE into 16-channel spatiotemporal patches
/// 3. The DiT processes patches as a sequence (like tokens in a language model)
/// 4. Full 3D attention captures spatial and temporal relationships simultaneously
/// 5. Flow matching denoises the video over scheduled timesteps
/// 6. The causal VAE decodes patches back to variable-duration video
///
/// Key characteristics:
/// - Native variable duration and resolution (no fixed grid)
/// - Trained on video + image data at native aspect ratios
/// - Full 3D attention (no factorization) for maximum quality
/// - "World simulator" approach for physically plausible generation
/// - 150 frames at 24 FPS by default (~6.25 seconds), up to 60s
///
/// When to use Sora:
/// - Highest-quality video generation
/// - Variable-length and multi-resolution content
/// - Physical simulation and world modeling
/// - Long-duration video generation
///
/// Limitations:
/// - Proprietary model (API-only access)
/// - Very large model with high compute requirements
/// - May struggle with complex physical interactions
/// - Generation can be slow for long videos
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT with 3D spatiotemporal patches
/// - Hidden dimension: 3072
/// - Transformer layers: 48
/// - Attention heads: 24
/// - Patch size: 2 (spatiotemporal)
/// - Latent channels: 16 (3D causal VAE)
/// - Context dimension: 4096 (CLIP + T5 dual encoder)
/// - VAE: 3D causal with 3 temporal layers
/// - Default: 150 frames at 24 FPS (~6.25 seconds)
/// - Training objective: Flow matching
/// - Native variable resolution and duration
///
/// Reference: OpenAI, "Video generation models as world simulators", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var sora = new SoraModel&lt;float&gt;();
///
/// // Generate video from text
/// var video = sora.GenerateFromText(
///     prompt: "A drone shot flying through a dense forest canopy at golden hour",
///     width: 1920,
///     height: 1080,
///     numFrames: 150,
///     fps: 24,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate video from an image
/// var animated = sora.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 150,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class SoraModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the 3D causal VAE (16).
    /// </summary>
    private const int LATENT_CHANNELS = 16;

    /// <summary>
    /// Hidden dimension of the DiT transformer (3072).
    /// </summary>
    private const int HIDDEN_DIM = 3072;

    /// <summary>
    /// Number of transformer layers (48).
    /// </summary>
    private const int NUM_LAYERS = 48;

    /// <summary>
    /// Number of attention heads (24).
    /// </summary>
    private const int NUM_HEADS = 24;

    /// <summary>
    /// Context dimension from the dual text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames (150, ~6.25 seconds at 24 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 150;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The DiT noise predictor with full 3D spatiotemporal attention.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The 3D causal VAE for spatiotemporal video compression.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The dual CLIP + T5 text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

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

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of SoraModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Sora defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 48-layer DiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the 3D causal VAE.</param>
    /// <param name="conditioner">Dual CLIP + T5 text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 150).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SoraModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
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
        _conditioner = conditioner;

        InitializeLayers(dit, temporalVAE, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the DiT and temporal VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for Sora defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for 3D causal VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default DiT:
    /// - 16 input/output channels, 3072 hidden dim, 48 layers, 24 heads
    /// - Patch size 2 for 3D spatiotemporal tokenization
    /// - Full 3D attention (no factorization)
    /// - 4096-dim context from dual text encoder
    ///
    /// Default 3D Causal VAE:
    /// - 16 latent channels, 3 temporal layers
    /// - Causal mode for autoregressive generation
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

        return new SoraModel<T>(
            dit: clonedDit,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
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
            Name = "Sora",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Sora-architecture DiT video generation with native spatiotemporal patches",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-spatiotemporal");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("variable_resolution", true);
        metadata.SetProperty("world_simulator", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
