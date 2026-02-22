using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// LTX-Video model for lightweight real-time video generation with extreme latent compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LTX-Video by Lightricks is designed for efficient video generation, using a lightweight
/// DiT transformer operating in a highly compressed latent space via a 3D causal VAE.
/// The extreme 192x compression ratio enables faster-than-real-time generation.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Lightweight DiT with 28 transformer layers and 1536 hidden dimension</description></item>
/// <item><description>16 attention heads for multi-head self-attention</description></item>
/// <item><description>3D causal VAE with 128 latent channels and 192x compression ratio</description></item>
/// <item><description>T5-XXL text encoder for 4096-dim context embeddings</description></item>
/// <item><description>Flow matching training objective for stable convergence</description></item>
/// <item><description>Patch size 1 (operates on individual latent tokens)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> LTX-Video generates videos faster than real-time by compressing heavily.
///
/// How LTX-Video works:
/// 1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
/// 2. Video is compressed by the 3D causal VAE into 128-channel latent (192x compression)
/// 3. The lightweight DiT denoises the latent using flow matching
/// 4. The causal VAE decodes the latent back to 720p video
///
/// Key characteristics:
/// - ~2B parameters (lightweight for a video model)
/// - 192x spatiotemporal compression (highest in class)
/// - 128 latent channels for rich representations despite compression
/// - 720p, 5 seconds at 24 FPS, faster-than-real-time generation
/// - Open-source with Lightricks weights
///
/// When to use LTX-Video:
/// - Real-time or interactive video generation
/// - Applications requiring low latency
/// - Edge deployment with limited compute
/// - Rapid prototyping and iteration
///
/// Limitations:
/// - High compression may lose fine spatial details
/// - Quality may not match larger models (Sora, HunyuanVideo)
/// - Best for medium-resolution content
/// - Trade-off between speed and fidelity
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Lightweight DiT with 3D causal VAE
/// - Hidden dimension: 1536
/// - Transformer layers: 28
/// - Attention heads: 16
/// - Patch size: 1 (direct latent token processing)
/// - Latent channels: 128 (extreme compression)
/// - Context dimension: 4096 (T5-XXL)
/// - VAE compression: 192x spatiotemporal
/// - Default: 121 frames at 24 FPS (~5 seconds)
/// - Training objective: Flow matching
/// - Total parameters: ~2B
///
/// Reference: Lightricks, "LTX-Video: Realtime Video Latent Diffusion", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var ltxVideo = new LTXVideoModel&lt;float&gt;();
///
/// // Generate video from text (faster than real-time)
/// var video = ltxVideo.GenerateFromText(
///     prompt: "A time-lapse of clouds moving across a mountain range",
///     width: 1280,
///     height: 720,
///     numFrames: 121,
///     fps: 24,
///     numInferenceSteps: 30,
///     guidanceScale: 7.0);
///
/// // Generate video from an image
/// var animated = ltxVideo.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 121,
///     numInferenceSteps: 30);
/// </code>
/// </example>
public class LTXVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the 3D causal VAE (128, extreme compression).
    /// </summary>
    /// <remarks>
    /// LTX-Video uses 128 latent channels to compensate for the extreme 192x compression,
    /// preserving enough information for high-quality reconstruction.
    /// </remarks>
    private const int LATENT_CHANNELS = 128;

    /// <summary>
    /// Hidden dimension of the lightweight DiT (1536).
    /// </summary>
    private const int HIDDEN_DIM = 1536;

    /// <summary>
    /// Number of transformer layers (28).
    /// </summary>
    private const int NUM_LAYERS = 28;

    /// <summary>
    /// Number of attention heads (16).
    /// </summary>
    private const int NUM_HEADS = 16;

    /// <summary>
    /// Context dimension from the T5-XXL text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size (1, operates on individual latent tokens).
    /// </summary>
    private const int PATCH_SIZE = 1;

    /// <summary>
    /// Default number of frames (121, ~5 seconds at 24 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 121;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The lightweight DiT noise predictor.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The 3D causal VAE with 192x compression.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The T5-XXL text encoder conditioning module.
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
    public override bool SupportsVideoToVideo => false;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of LTXVideoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses LTX-Video defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 28-layer lightweight DiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the 3D causal VAE with 192x compression.</param>
    /// <param name="conditioner">T5-XXL text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 121).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public LTXVideoModel(
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
    /// <param name="dit">Custom DiT predictor, or null for LTX-Video defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for 3D causal VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default Lightweight DiT:
    /// - 128 input/output channels, 1536 hidden dim, 28 layers, 16 heads
    /// - Patch size 1 (direct latent token processing)
    /// - 4096-dim context from T5-XXL
    ///
    /// Default 3D Causal VAE:
    /// - 192x spatiotemporal compression
    /// - 128 latent channels for rich representation
    /// - 2 temporal layers, causal mode enabled
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
            numTemporalLayers: 2,
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

        return new LTXVideoModel<T>(
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
            Name = "LTX-Video",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "LTX-Video lightweight real-time video generation with extreme latent compression",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "lightweight-dit-3d-causal-vae");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("compression_ratio", 192);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("faster_than_realtime", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
