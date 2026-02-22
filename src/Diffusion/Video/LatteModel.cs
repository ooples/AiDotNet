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
/// Latte model for Latent Diffusion Transformer video generation with factorized spatial-temporal attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Latte applies Diffusion Transformer (DiT) architecture to video generation by exploring
/// factorized spatial-temporal attention patterns within transformer blocks. The model
/// decomposes full 3D attention into efficient spatial and temporal components.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>DiT backbone with 28 transformer layers and 1152 hidden dimension</description></item>
/// <item><description>4 attention variants: joint, spatial-first, temporal-first, decomposed</description></item>
/// <item><description>16 attention heads with efficient O(n) factorized attention</description></item>
/// <item><description>T5-XXL text encoder for 4096-dim context embeddings</description></item>
/// <item><description>Standard SD VAE for per-frame spatial compression (4 latent channels)</description></item>
/// <item><description>Patch size 2 for spatiotemporal tokenization</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Latte applies DiT (Diffusion Transformer) concepts to video generation.
///
/// How Latte works:
/// 1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
/// 2. Each video frame is encoded by the SD VAE into 4 latent channels
/// 3. Latent frames are patchified and processed by the DiT with factorized attention
/// 4. Spatial attention handles per-frame content, temporal attention handles motion
/// 5. The VAE decodes each latent frame back to pixel space
///
/// Key characteristics:
/// - Explores 4 attention decomposition strategies for efficiency
/// - "Decomposed" variant (spatial then temporal) achieves best quality/speed
/// - Uses per-frame VAE (standard SD VAE, not temporal)
/// - 16 frames at 8 FPS by default (~2 seconds)
/// - ~700M parameter DiT backbone
///
/// When to use Latte:
/// - Research on efficient video DiT architectures
/// - Short clip generation from text prompts
/// - Exploring spatial-temporal attention decomposition
/// - Lightweight alternative to full 3D attention models
///
/// Limitations:
/// - Per-frame VAE may cause minor temporal artifacts
/// - Shorter duration than temporal-VAE-based models
/// - Lower resolution than SDXL-based video models
/// - Research model, not production-grade quality
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT with factorized spatial-temporal attention
/// - Hidden dimension: 1152
/// - Transformer layers: 28
/// - Attention heads: 16
/// - Patch size: 2
/// - Latent channels: 4 (standard SD VAE)
/// - Context dimension: 4096 (T5-XXL)
/// - Default: 16 frames at 8 FPS (~2 seconds)
/// - Noise schedule: Linear beta [0.0001, 0.02], 1000 timesteps
/// - Scheduler: DDIM
/// - Parameters: ~700M (DiT backbone)
///
/// Reference: Ma et al., "Latte: Latent Diffusion Transformer for Video Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var latte = new LatteModel&lt;float&gt;();
///
/// // Generate video from text
/// var video = latte.GenerateFromText(
///     prompt: "A cat sitting on a windowsill watching birds",
///     width: 512,
///     height: 512,
///     numFrames: 16,
///     fps: 8,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
/// </code>
/// </example>
public class LatteModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the standard SD VAE (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Hidden dimension of the DiT transformer (1152).
    /// </summary>
    private const int HIDDEN_DIM = 1152;

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
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames (16).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 16;

    /// <summary>
    /// Default frames per second (8).
    /// </summary>
    private const int DEFAULT_FPS = 8;

    #endregion

    #region Fields

    /// <summary>
    /// The DiT noise predictor with factorized spatial-temporal attention.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The standard SD VAE for per-frame spatial compression.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The T5-XXL text encoder conditioning module.
    /// </summary>
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
    public override bool SupportsImageToVideo => false;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of LatteModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Latte defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDIM scheduler for efficient inference.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 28-layer factorized DiT.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD VAE.</param>
    /// <param name="conditioner">T5-XXL text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 16).</param>
    /// <param name="defaultFPS">Default frames per second (default: 8).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public LatteModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        StandardVAE<T>? vae = null,
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
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(dit, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the DiT and VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for Latte defaults.</param>
    /// <param name="vae">Custom VAE, or null for standard SD VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default DiT:
    /// - 4 input/output channels, 1152 hidden dim, 28 layers, 16 heads
    /// - Patch size 2 for spatiotemporal tokenization
    /// - 4096-dim context from T5-XXL
    ///
    /// Default VAE:
    /// - Standard SD VAE with 4 latent channels
    /// - 128 base channels, [1,2,4,4] multipliers
    /// - 0.18215 latent scale factor
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        StandardVAE<T>? vae,
        int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
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
        var vaeParams = _vae.GetParameters();

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
        var vaeCount = _vae.ParameterCount;

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
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new LatteModel<T>(
            dit: clonedDit,
            vae: new StandardVAE<T>(
                inputChannels: 3,
                latentChannels: LATENT_CHANNELS,
                baseChannels: 128,
                channelMultipliers: new[] { 1, 2, 4, 4 },
                numResBlocksPerLevel: 2,
                latentScaleFactor: 0.18215),
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
            Name = "Latte",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Latte latent diffusion transformer for video generation with factorized attention",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-factorized-st-attention");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("attention_variants", 4);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
