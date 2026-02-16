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
/// Open-Sora model for open-source Sora-like video generation with STDiT architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Open-Sora is an open-source reproduction of Sora-like capabilities using the
/// Spatial-Temporal DiT (STDiT) architecture. It features efficient spatial-temporal
/// attention factorization, rectified flow training, and multi-resolution masking strategy.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>STDiT (Spatial-Temporal Diffusion Transformer) with 28 layers</description></item>
/// <item><description>1152 hidden dimension with 16 attention heads</description></item>
/// <item><description>3D causal VAE with 4 latent channels for spatiotemporal compression</description></item>
/// <item><description>T5-XXL text encoder for 4096-dim context embeddings</description></item>
/// <item><description>Rectified flow training objective for efficiency</description></item>
/// <item><description>Multi-resolution training with masking strategy</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Open-Sora is an open-source alternative to OpenAI's Sora.
///
/// How Open-Sora works:
/// 1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
/// 2. Video is compressed by the 3D causal VAE into 4-channel latent space
/// 3. The STDiT applies spatial attention first, then temporal attention per layer
/// 4. Rectified flow denoises the video latent over scheduled timesteps
/// 5. The causal VAE decodes the latent back to video frames
///
/// Key characteristics:
/// - Open-source community project replicating Sora capabilities
/// - STDiT factorizes spatial and temporal attention for efficiency
/// - Rectified flow training (straighter trajectories than DDPM)
/// - Multi-resolution training enables variable aspect ratio/duration
/// - 51 frames at 24 FPS by default (~2.1 seconds)
///
/// When to use Open-Sora:
/// - Open-source video generation research
/// - Variable-resolution video generation
/// - Text-to-video, image-to-video, and video-to-video tasks
/// - Community-driven experimentation and fine-tuning
///
/// Limitations:
/// - Quality below commercial models (Sora, Veo)
/// - Research-stage model with ongoing improvements
/// - Limited duration compared to larger models
/// - Requires significant compute for training
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: STDiT (Spatial-Temporal Diffusion Transformer)
/// - Hidden dimension: 1152
/// - Transformer layers: 28
/// - Attention heads: 16
/// - Patch size: 2
/// - Latent channels: 4 (3D causal VAE)
/// - Context dimension: 4096 (T5-XXL)
/// - Training objective: Rectified flow
/// - Multi-resolution training with masking
/// - Default: 51 frames at 24 FPS (~2.1 seconds)
/// - Supports: text-to-video, image-to-video, video-to-video
///
/// Reference: Zheng et al., "Open-Sora: Democratizing Efficient Video Production for All", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var openSora = new OpenSoraModel&lt;float&gt;();
///
/// // Generate video from text
/// var video = openSora.GenerateFromText(
///     prompt: "A serene lake reflecting autumn foliage with gentle ripples",
///     width: 512,
///     height: 512,
///     numFrames: 51,
///     fps: 24,
///     numInferenceSteps: 30,
///     guidanceScale: 7.0);
///
/// // Generate video from an image
/// var animated = openSora.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 51,
///     numInferenceSteps: 30);
/// </code>
/// </example>
public class OpenSoraModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the 3D causal VAE (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Hidden dimension of the STDiT transformer (1152).
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
    /// Default number of frames (51, ~2.1 seconds at 24 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 51;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The STDiT noise predictor with spatial-temporal attention factorization.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The 3D causal VAE for spatiotemporal video compression.
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
    public override bool SupportsVideoToVideo => true;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of OpenSoraModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Open-Sora defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the rectified flow objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 28-layer STDiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the 3D causal VAE.</param>
    /// <param name="conditioner">T5-XXL text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 51).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public OpenSoraModel(
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
    /// Initializes the STDiT and temporal VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for Open-Sora defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for 3D causal VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default STDiT:
    /// - 4 input/output channels, 1152 hidden dim, 28 layers, 16 heads
    /// - Patch size 2, spatial-temporal factorized attention
    /// - 4096-dim context from T5-XXL
    ///
    /// Default 3D Causal VAE:
    /// - 4 latent channels, causal temporal processing
    /// - 2 temporal layers with kernel size 3
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

        return new OpenSoraModel<T>(
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
            Name = "Open-Sora",
            Version = "1.2",
            ModelType = ModelType.NeuralNetwork,
            Description = "Open-Sora STDiT video generation with rectified flow training",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "stdit");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("training_objective", "rectified-flow");
        metadata.SetProperty("multi_resolution", true);
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
