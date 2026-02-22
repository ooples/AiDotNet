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
/// Kling model — 3D spatiotemporal attention video generation by Kuaishou.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kling uses a 3D VAE with full spatiotemporal attention for high-quality video generation
/// with strong motion consistency and physics understanding. It supports up to 2 minutes of
/// video at 1080p resolution.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>DiT backbone with 3D full attention (36 layers, 2048 hidden dim)</description></item>
/// <item><description>Temporal VAE for spatiotemporal compression (causal mode)</description></item>
/// <item><description>Large-scale text encoder (4096-dim context)</description></item>
/// <item><description>16-channel latent space for high-fidelity reconstruction</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Kling is a video generation model from Kuaishou that creates
/// high-quality videos from text or images.
///
/// How Kling works:
/// 1. Text prompt is encoded into a 4096-dimensional embedding
/// 2. A 3D DiT transformer generates video in compressed latent space
/// 3. Full 3D attention (not factorized) captures spatial AND temporal relationships
/// 4. A temporal VAE decompresses the latent video into pixel-space frames
///
/// Advantages:
/// - Very high video quality at 1080p resolution
/// - Long video support (up to 2 minutes)
/// - Strong physics and motion understanding
/// - Both text-to-video and image-to-video generation
///
/// Limitations:
/// - Very large model requiring significant GPU memory
/// - Proprietary architecture with limited public details
/// - Slower generation than smaller video models
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT with full 3D spatiotemporal attention
/// - Parameters: ~5B+ (estimated)
/// - Backbone: 36 transformer layers, 2048 hidden dim, 16 attention heads
/// - Latent space: 16 channels with temporal VAE
/// - Text encoder: 4096-dimensional context embedding
/// - Max resolution: 1080p (1920x1080)
/// - Max duration: 2 minutes at 30 FPS
/// - Noise schedule: Flow matching
///
/// Reference: Kuaishou, "Kling: A Text-to-Video Generation Model", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with default settings
/// var kling = new KlingModel&lt;float&gt;();
///
/// // Generate video from text
/// var frames = kling.GenerateFromText(
///     prompt: "A drone flying over a mountain landscape at sunset",
///     numFrames: 60,
///     numInferenceSteps: 50,
///     guidanceScale: 7.0);
///
/// // Generate video from an image
/// var animatedFrames = kling.GenerateFromImage(
///     inputImage: referenceImage,
///     numFrames: 30);
/// </code>
/// </example>
public class KlingModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels for Kling's temporal VAE (16 for high fidelity).
    /// </summary>
    private const int LATENT_CHANNELS = 16;

    /// <summary>
    /// Hidden dimension of the DiT backbone (2048).
    /// </summary>
    private const int HIDDEN_DIM = 2048;

    /// <summary>
    /// Number of transformer layers in the DiT backbone (36).
    /// </summary>
    private const int NUM_LAYERS = 36;

    /// <summary>
    /// Number of attention heads in the DiT backbone (16).
    /// </summary>
    private const int NUM_HEADS = 16;

    /// <summary>
    /// Context dimension from the text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Default guidance scale for Kling (7.0).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.0;

    #endregion

    #region Fields

    private DiTNoisePredictor<T> _dit;
    private TemporalVAE<T> _temporalVAE;
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
    /// Initializes a new instance of KlingModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Diffusion model options. If null, uses Kling defaults.</param>
    /// <param name="scheduler">Noise scheduler. If null, uses flow matching scheduler.</param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates Kling's default.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates Kling's default.</param>
    /// <param name="conditioner">Text conditioning module.</param>
    /// <param name="defaultNumFrames">Default number of frames to generate (default: 150).</param>
    /// <param name="defaultFPS">Default frames per second (default: 30).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public KlingModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 150,
        int defaultFPS = 30,
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
            patchSize: 2,
            contextDim: CONTEXT_DIM,
            seed: seed);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3,
            temporalKernelSize: 3,
            causalMode: true,
            latentScaleFactor: 0.13025,
            seed: seed);
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

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();
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
        var ditCount = _dit.ParameterCount;
        var vaeCount = _temporalVAE.GetParameters().Length;

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
        _temporalVAE.SetParameters(vaeParams);
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
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: 2,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new KlingModel<T>(
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
            Name = "Kling",
            Version = "1.5",
            ModelType = ModelType.NeuralNetwork,
            Description = "Kling 3D spatiotemporal attention video generation model by Kuaishou",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "3d-dit-full-attention");
        metadata.SetProperty("backbone", "DiT-36L-2048H");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("max_duration_seconds", 120);
        metadata.SetProperty("max_resolution", "1080p");
        metadata.SetProperty("noise_schedule", "flow_matching");

        return metadata;
    }

    #endregion
}
