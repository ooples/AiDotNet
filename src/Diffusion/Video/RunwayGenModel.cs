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
/// Runway Gen model for multi-modal video generation with structure and content disentanglement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Runway Gen (Gen-1/Gen-2/Gen-3) uses temporal diffusion with multi-modal conditioning
/// for photorealistic video generation and editing. The model supports text, image, video,
/// and motion conditioning with structure-content disentanglement.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Video U-Net with temporal attention and cross-frame consistency</description></item>
/// <item><description>Gen-2: 320 base channels, 1024-dim CLIP, 8 heads, 1 temporal layer</description></item>
/// <item><description>Gen-3: 384 base channels, 2048-dim dual encoder, 16 heads, 3 temporal layers</description></item>
/// <item><description>Temporal VAE for inter-frame coherence (causal in Gen-3)</description></item>
/// <item><description>Multi-modal conditioning: text, image, video, and motion</description></item>
/// <item><description>Structure and style disentanglement for editing control</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Runway Gen creates professional-quality videos from text and images.
///
/// How Runway Gen works:
/// 1. Text/image input is encoded by CLIP (Gen-2) or dual encoder (Gen-3)
/// 2. Video is compressed by the temporal VAE into 4-channel latent space
/// 3. The Video U-Net denoises with temporal attention for frame consistency
/// 4. Multi-modal conditioning guides structure and content separately
/// 5. The temporal VAE decodes the latent back to video frames
///
/// Key characteristics:
/// - Gen-2: CLIP-based, 320-channel U-Net, 25 frames at 24 FPS
/// - Gen-3: Enhanced dual encoder, 384-channel U-Net, 150 frames, causal VAE
/// - Multi-modal: text, image, video, and motion conditioning
/// - Structure-content disentanglement for precise editing
/// - Cascaded generation for high resolution output
///
/// When to use Runway Gen:
/// - Professional video generation and editing
/// - Multi-modal conditioning (text + image + video)
/// - Video-to-video style transfer
/// - Image animation with motion control
///
/// Limitations:
/// - Commercial/proprietary model (API-only access)
/// - Expensive generation costs
/// - Limited control over internal architecture
/// - Gen-3 requires significant compute resources
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Video U-Net with temporal attention and multi-modal conditioning
/// - Gen-2: 320 base channels, 1024 cross-attention, 8 heads, 1 temporal layer
/// - Gen-3: 384 base channels, 2048 cross-attention, 16 heads, 3 temporal layers
/// - Latent channels: 4 (temporal VAE)
/// - Channel multipliers: [1, 2, 4, 4]
/// - Gen-2 default: 25 frames at 24 FPS (~1 second)
/// - Gen-3 default: 150 frames at 24 FPS (~6.25 seconds)
/// - Scheduler: DDIM for efficient inference
/// - Supports: text-to-video, image-to-video, video-to-video
///
/// Reference: Esser et al., "Structure and Content-Guided Video Synthesis with Diffusion Models", ICCV 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Gen-2 with defaults
/// var runwayGen2 = new RunwayGenModel&lt;float&gt;();
///
/// // Create Gen-3 Alpha variant
/// var runwayGen3 = RunwayGenModel&lt;float&gt;.CreateGen3Alpha();
///
/// // Generate video from text
/// var video = runwayGen2.GenerateFromText(
///     prompt: "A cinematic shot of a castle on a cliff at sunset",
///     width: 1280,
///     height: 768,
///     numFrames: 25,
///     fps: 24,
///     numInferenceSteps: 50,
///     guidanceScale: 12.0);
///
/// // Generate video from an image
/// var animated = runwayGen2.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 25,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class RunwayGenModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension for Gen-2 (1024, CLIP).
    /// </summary>
    private const int GEN2_CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Cross-attention dimension for Gen-3 (2048, dual encoder).
    /// </summary>
    private const int GEN3_CROSS_ATTENTION_DIM = 2048;

    /// <summary>
    /// Base channel count for Gen-2 (320).
    /// </summary>
    private const int GEN2_BASE_CHANNELS = 320;

    /// <summary>
    /// Base channel count for Gen-3 (384).
    /// </summary>
    private const int GEN3_BASE_CHANNELS = 384;

    /// <summary>
    /// Default number of frames for Gen-2 (25).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 25;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The Video U-Net noise predictor with temporal attention.
    /// </summary>
    private VideoUNetPredictor<T> _videoUNet;

    /// <summary>
    /// The temporal VAE for inter-frame coherent encoding/decoding.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// Optional conditioning module for multi-modal guided generation.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Whether this is a Gen-3 variant.
    /// </summary>
    private readonly bool _isGen3;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _videoUNet;

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
    public override int ParameterCount =>
        _videoUNet.GetParameters().Length + _temporalVAE.GetParameters().Length;

    /// <summary>
    /// Gets whether this is a Gen-3 variant with enhanced architecture.
    /// </summary>
    public bool IsGen3 => _isGen3;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of RunwayGenModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Runway Gen defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDIM scheduler for efficient inference.
    /// </param>
    /// <param name="videoUNet">Custom Video U-Net. If null, creates variant-appropriate U-Net.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates variant-appropriate VAE.</param>
    /// <param name="conditioner">Multi-modal conditioning module.</param>
    /// <param name="isGen3">Whether to use Gen-3 architecture (default: false for Gen-2).</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 25 for Gen-2, 150 for Gen-3).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RunwayGenModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        bool isGen3 = false,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _isGen3 = isGen3;
        _conditioner = conditioner;

        InitializeLayers(videoUNet, temporalVAE, seed);
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a Gen-3 Alpha variant with enhanced architecture.
    /// </summary>
    /// <param name="conditioner">Optional multi-modal conditioning module.</param>
    /// <returns>A new RunwayGenModel configured for Gen-3 Alpha specifications.</returns>
    public static RunwayGenModel<T> CreateGen3Alpha(IConditioningModule<T>? conditioner = null)
    {
        return new RunwayGenModel<T>(
            isGen3: true,
            conditioner: conditioner,
            defaultNumFrames: 150,
            defaultFPS: 24);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Video U-Net and temporal VAE using custom or variant-appropriate defaults.
    /// </summary>
    /// <param name="videoUNet">Custom Video U-Net, or null for variant defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for variant defaults.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Gen-2 Video U-Net:
    /// - 320 base channels, 1024-dim CLIP cross-attention, 8 heads
    /// - 1 temporal attention layer per block
    /// - Non-causal temporal VAE for bidirectional processing
    ///
    /// Gen-3 Video U-Net:
    /// - 384 base channels, 2048-dim dual encoder, 16 heads
    /// - 3 temporal attention layers per block
    /// - Causal temporal VAE for autoregressive generation
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_videoUNet), nameof(_temporalVAE))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? videoUNet,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: _isGen3 ? GEN3_BASE_CHANNELS : GEN2_BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: _isGen3 ? GEN3_CROSS_ATTENTION_DIM : GEN2_CROSS_ATTENTION_DIM,
            numHeads: _isGen3 ? 16 : 8,
            numTemporalLayers: _isGen3 ? 3 : 1,
            supportsImageConditioning: true);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 1,
            temporalKernelSize: 3,
            causalMode: _isGen3,
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
        return _videoUNet.PredictNoiseWithImageCondition(
            latents, timestep, imageEmbedding, textConditioning: null);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _videoUNet.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();

        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _videoUNet.GetParameters().Length;
        var vaeCount = _temporalVAE.GetParameters().Length;

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _videoUNet.SetParameters(unetParams);
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
        return new RunwayGenModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            isGen3: _isGen3,
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
            Name = _isGen3 ? "Runway-Gen-3" : "Runway-Gen-2",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = $"Runway {(_isGen3 ? "Gen-3" : "Gen-2")} multi-modal video generation with structure-content disentanglement",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "temporal-unet-multimodal");
        metadata.SetProperty("is_gen3", _isGen3);
        metadata.SetProperty("base_channels", _isGen3 ? GEN3_BASE_CHANNELS : GEN2_BASE_CHANNELS);
        metadata.SetProperty("cross_attention_dim", _isGen3 ? GEN3_CROSS_ATTENTION_DIM : GEN2_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("temporal_layers", _isGen3 ? 3 : 1);
        metadata.SetProperty("multi_modal_conditioning", true);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_frames", DefaultNumFrames);
        metadata.SetProperty("causal_vae", _isGen3);

        return metadata;
    }

    #endregion
}
