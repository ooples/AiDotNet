using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// Wan video model for Alibaba's scalable DiT video generation with full 3D attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wan by Alibaba uses a scalable DiT architecture with full 3D attention (no factorization)
/// and a specialized WanVAE for temporally compressed video generation. The model supports
/// multiple scale variants: 1.3B, 5B, and 14B parameters.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Scalable DiT with full 3D attention (no spatial-temporal factorization)</description></item>
/// <item><description>1.3B variant: 1536 hidden, 30 layers, 12 heads</description></item>
/// <item><description>5B variant: 2560 hidden, 36 layers, 20 heads</description></item>
/// <item><description>14B variant: 3072 hidden, 40 layers, 24 heads (default)</description></item>
/// <item><description>WanVAE: specialized causal 3D VAE with 16 latent channels</description></item>
/// <item><description>T5-XXL text encoder for 4096-dim context embeddings</description></item>
/// <item><description>Flow matching training objective</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Wan generates high-quality videos with multiple size variants.
///
/// How Wan works:
/// 1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
/// 2. Video is compressed by WanVAE into 16-channel latent with causal temporal compression
/// 3. The scalable DiT processes all patches with full 3D attention (no shortcuts)
/// 4. Full 3D attention captures all spatial-temporal relationships simultaneously
/// 5. Flow matching denoises the video over scheduled timesteps
/// 6. WanVAE decodes the latent back to video frames
///
/// Key characteristics:
/// - Three scale variants: 1.3B (fast), 5B (balanced), 14B (highest quality)
/// - Full 3D attention (no factorization) for maximum quality at each scale
/// - WanVAE: specialized causal 3D VAE with 3 temporal layers
/// - Text-to-video and image-to-video support
/// - Open-source weights available
/// - 81 frames at 16 FPS by default (~5 seconds)
///
/// When to use Wan:
/// - Scalable video generation (choose variant for quality/speed trade-off)
/// - High-quality text-to-video generation
/// - Image animation (image-to-video)
/// - Research on scalable video architectures
///
/// Limitations:
/// - 14B variant requires significant GPU memory
/// - 1.3B variant sacrifices quality for speed
/// - Full 3D attention is O(n^2) in sequence length
/// - Limited video duration compared to cascaded models
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Scalable DiT with full 3D attention
/// - 1.3B: 1536 hidden, 30 layers, 12 heads (~1.3B parameters)
/// - 5B: 2560 hidden, 36 layers, 20 heads (~5B parameters)
/// - 14B: 3072 hidden, 40 layers, 24 heads (~14B parameters)
/// - Patch size: 2 (all variants)
/// - Latent channels: 16 (WanVAE)
/// - Context dimension: 4096 (T5-XXL)
/// - WanVAE: causal 3D VAE, 3 temporal layers, kernel size 3
/// - Default: 81 frames at 16 FPS (~5 seconds)
/// - Training objective: Flow matching
/// - Open-source: Yes
///
/// Reference: Alibaba, "Wan: Open and Advanced Large-Scale Video Generative Models", 2025
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create default 14B variant
/// var wan = new WanVideoModel&lt;float&gt;();
///
/// // Create lightweight 1.3B variant
/// var wanLight = WanVideoModel&lt;float&gt;.Create1_3B();
///
/// // Create balanced 5B variant
/// var wanMedium = WanVideoModel&lt;float&gt;.Create5B();
///
/// // Generate video from text
/// var video = wan.GenerateFromText(
///     prompt: "A traditional Chinese ink painting coming to life with gentle motion",
///     width: 1280,
///     height: 720,
///     numFrames: 81,
///     fps: 16,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate video from an image
/// var animated = wan.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 81,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class WanVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the WanVAE (16).
    /// </summary>
    private const int LATENT_CHANNELS = 16;

    /// <summary>
    /// Context dimension from the T5-XXL text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames (81, ~5 seconds at 16 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 81;

    /// <summary>
    /// Default frames per second (16).
    /// </summary>
    private const int DEFAULT_FPS = 16;

    #endregion

    #region Fields

    /// <summary>
    /// The scalable DiT noise predictor with full 3D attention.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The WanVAE (causal 3D VAE) for temporally compressed video encoding/decoding.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The T5-XXL text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// The model variant identifier (1.3B, 5B, or 14B).
    /// </summary>
    private readonly string _variant;

    /// <summary>
    /// Number of attention heads for the current variant.
    /// </summary>
    private readonly int _numHeads;

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

    /// <summary>
    /// Gets the model variant identifier (1.3B, 5B, or 14B).
    /// </summary>
    public string Variant => _variant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of WanVideoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Wan defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates variant-appropriate DiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the WanVAE.</param>
    /// <param name="conditioner">T5-XXL text encoder conditioning module.</param>
    /// <param name="variant">Model variant: "1.3B", "5B", or "14B" (default: "14B").</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 81).</param>
    /// <param name="defaultFPS">Default frames per second (default: 16).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public WanVideoModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        string variant = "14B",
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
        _variant = variant;
        _conditioner = conditioner;

        var (_, _, numHeads) = GetVariantConfig(variant);
        _numHeads = numHeads;

        InitializeLayers(dit, temporalVAE, seed);
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a 1.3B lightweight variant for fast generation.
    /// </summary>
    /// <param name="conditioner">Optional T5-XXL text encoder conditioning module.</param>
    /// <returns>A new WanVideoModel configured for 1.3B specifications.</returns>
    public static WanVideoModel<T> Create1_3B(IConditioningModule<T>? conditioner = null)
    {
        return new WanVideoModel<T>(variant: "1.3B", conditioner: conditioner);
    }

    /// <summary>
    /// Creates a 5B medium variant balancing quality and speed.
    /// </summary>
    /// <param name="conditioner">Optional T5-XXL text encoder conditioning module.</param>
    /// <returns>A new WanVideoModel configured for 5B specifications.</returns>
    public static WanVideoModel<T> Create5B(IConditioningModule<T>? conditioner = null)
    {
        return new WanVideoModel<T>(variant: "5B", conditioner: conditioner);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Gets the architecture configuration for a given variant.
    /// </summary>
    private static (int hiddenDim, int numLayers, int numHeads) GetVariantConfig(string variant)
    {
        return variant switch
        {
            "1.3B" => (1536, 30, 12),
            "5B" => (2560, 36, 20),
            _ => (3072, 40, 24),
        };
    }

    /// <summary>
    /// Initializes the DiT and WanVAE layers using custom or variant-appropriate defaults.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for variant defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for WanVAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default DiT (varies by variant):
    /// - 1.3B: 1536 hidden, 30 layers, 12 heads
    /// - 5B: 2560 hidden, 36 layers, 20 heads
    /// - 14B: 3072 hidden, 40 layers, 24 heads
    /// - All: 16 latent channels, patch size 2, 4096-dim context
    ///
    /// Default WanVAE:
    /// - 16 latent channels, causal 3D VAE
    /// - 3 temporal layers with kernel size 3
    /// - 0.13025 latent scale factor
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_dit), nameof(_temporalVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        var (hiddenDim, numLayers, numHeads) = GetVariantConfig(_variant);

        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: hiddenDim,
            numLayers: numLayers,
            numHeads: numHeads,
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
        var (hiddenDim, numLayers, numHeads) = GetVariantConfig(_variant);

        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: hiddenDim,
            numLayers: numLayers,
            numHeads: numHeads,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new WanVideoModel<T>(
            dit: clonedDit,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            variant: _variant,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var (hiddenDim, numLayers, numHeads) = GetVariantConfig(_variant);

        var metadata = new ModelMetadata<T>
        {
            Name = $"Wan-{_variant}",
            Version = "2.1",
            ModelType = ModelType.NeuralNetwork,
            Description = $"Wan {_variant} video generation with full 3D attention and WanVAE",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-full-3d-attention");
        metadata.SetProperty("variant", _variant);
        metadata.SetProperty("hidden_dim", hiddenDim);
        metadata.SetProperty("num_layers", numLayers);
        metadata.SetProperty("num_heads", numHeads);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
