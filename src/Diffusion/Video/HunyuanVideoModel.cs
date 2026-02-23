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
/// HunyuanVideo model for dual-stream DiT video generation with unified image-video capability.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HunyuanVideo by Tencent uses a "Dual-stream to Single-stream" Diffusion Transformer
/// (DS-DiT) architecture with a 3D causal VAE for high-resolution video generation.
/// The dual-stream design processes text and video tokens separately in early layers,
/// then merges them for joint attention in later layers.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>DS-DiT (Dual-stream to Single-stream DiT) with 40 transformer layers</description></item>
/// <item><description>3072 hidden dimension with 24 attention heads</description></item>
/// <item><description>3D causal VAE with 4x4x4 spatiotemporal compression (16 latent channels)</description></item>
/// <item><description>MLLM-based text encoder for rich semantic understanding (4096-dim context)</description></item>
/// <item><description>Flow matching training objective for stable convergence</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> HunyuanVideo is Tencent's open-source video generation model.
///
/// How HunyuanVideo works:
/// 1. Text prompt is encoded by an MLLM-based encoder into 4096-dim embeddings
/// 2. Video is compressed by the 3D causal VAE into a 16-channel latent (4x4x4 compression)
/// 3. The DS-DiT processes text and video in dual streams, then merges for joint attention
/// 4. Flow matching denoises the video latent over the scheduled timesteps
/// 5. The causal VAE decodes the latent back to 720p video
///
/// Key characteristics:
/// - 13B parameters total (one of the largest open-source video models)
/// - 720p resolution, 5+ second duration, 129 frames at 24 FPS
/// - Open-source weights available
/// - Unified image-video generation (both supported)
/// - 3D causal VAE enables temporal consistency
///
/// When to use HunyuanVideo:
/// - High-quality open-source video generation
/// - Text-to-video with strong prompt adherence
/// - Image-to-video animation
/// - Research and experimentation with large video models
///
/// Limitations:
/// - Very large model (13B parameters, requires significant GPU memory)
/// - Slower generation than lightweight models
/// - Causal VAE may have slight quality loss at clip boundaries
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DS-DiT (Dual-stream to Single-stream Diffusion Transformer)
/// - Hidden dimension: 3072
/// - Transformer layers: 40 (dual-stream early, single-stream late)
/// - Attention heads: 24
/// - Patch size: 2x2 spatiotemporal patches
/// - Latent channels: 16 (compressed by 3D causal VAE)
/// - Context dimension: 4096 (MLLM text encoder)
/// - VAE compression: 4x4x4 spatiotemporal
/// - Default: 129 frames at 24 FPS (~5.4 seconds)
/// - Training objective: Flow matching
/// - Total parameters: ~13B
///
/// Reference: Kong et al., "HunyuanVideo: A Systematic Framework For Large Video Generative Models", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var hunyuan = new HunyuanVideoModel&lt;float&gt;();
///
/// // Generate video from text (requires conditioner)
/// var video = hunyuan.GenerateFromText(
///     prompt: "A golden retriever running through a sunlit meadow",
///     width: 1280,
///     height: 720,
///     numFrames: 129,
///     fps: 24,
///     numInferenceSteps: 50,
///     guidanceScale: 6.0);
///
/// // Generate video from an image
/// var animated = hunyuan.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 129,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class HunyuanVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the 3D causal VAE (16).
    /// </summary>
    /// <remarks>
    /// HunyuanVideo's 3D causal VAE uses 16 latent channels (4x more than SD's 4),
    /// providing richer latent representations for temporal information.
    /// </remarks>
    private const int LATENT_CHANNELS = 16;

    /// <summary>
    /// Hidden dimension of the DS-DiT transformer (3072).
    /// </summary>
    private const int HIDDEN_DIM = 3072;

    /// <summary>
    /// Number of transformer layers in the DS-DiT (40).
    /// </summary>
    private const int NUM_LAYERS = 40;

    /// <summary>
    /// Number of attention heads (24).
    /// </summary>
    private const int NUM_HEADS = 24;

    /// <summary>
    /// Context dimension from the MLLM text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames (129, ~5.4 seconds at 24 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 129;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The DS-DiT noise predictor with dual-stream to single-stream architecture.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The 3D causal VAE for spatiotemporal video compression.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The MLLM-based text encoder conditioning module.
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
    /// Initializes a new instance of HunyuanVideoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses HunyuanVideo defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the standard 40-layer DS-DiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the 3D causal VAE.</param>
    /// <param name="conditioner">MLLM text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 129).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public HunyuanVideoModel(
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
    /// <param name="dit">Custom DiT predictor, or null for HunyuanVideo defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for 3D causal VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default DS-DiT:
    /// - 16 input/output channels, 3072 hidden dim, 40 layers, 24 heads
    /// - Patch size 2 for spatiotemporal tokenization
    /// - 4096-dim context from MLLM text encoder
    ///
    /// Default 3D Causal VAE:
    /// - 4x4x4 spatiotemporal compression
    /// - 16 latent channels, causal temporal processing
    /// - 3 temporal layers with kernel size 3
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

        return new HunyuanVideoModel<T>(
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
            Name = "HunyuanVideo",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "HunyuanVideo dual-stream DiT video generation with 3D causal VAE",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "ds-dit-3d-causal-vae");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("parameters_billions", 13);
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
