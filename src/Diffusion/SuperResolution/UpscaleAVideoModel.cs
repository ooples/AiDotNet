using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// Upscale-A-Video model for temporally consistent video super-resolution with diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Upscale-A-Video extends image super-resolution to video with temporal consistency,
/// using temporal attention layers and flow-guided recurrent propagation to achieve
/// flicker-free 4x upscaling of video content.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Video U-Net with temporal attention and temporal convolutions</description></item>
/// <item><description>8 input channels (4 latent + 4 downscaled low-res conditioning)</description></item>
/// <item><description>2 temporal attention layers per block for inter-frame consistency</description></item>
/// <item><description>Temporal VAE for temporally coherent encoding/decoding</description></item>
/// <item><description>Flow-guided recurrent propagation for long-range temporal consistency</description></item>
/// <item><description>DDIM scheduler for efficient inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Upscale-A-Video increases video resolution by 4x without flickering.
///
/// How Upscale-A-Video works:
/// 1. Each frame is encoded with a temporal VAE that considers neighboring frames
/// 2. The low-resolution video is concatenated with latent noise (8 input channels)
/// 3. Temporal attention ensures frames are consistent with each other
/// 4. Flow-guided propagation maintains consistency across long sequences
/// 5. The temporal VAE decodes the result to high-resolution flicker-free video
///
/// Key characteristics:
/// - 4x video upscaling with temporal consistency
/// - Temporal attention layers prevent inter-frame flickering
/// - Flow-guided recurrent propagation for long-range coherence
/// - Built on SD architecture with temporal extensions
/// - Processes 16 frames at a time at 24 FPS by default
///
/// When to use Upscale-A-Video:
/// - Upscaling low-resolution video recordings
/// - Enhancing video quality for streaming/display
/// - Restoring old or compressed video footage
/// - Improving AI-generated video resolution
///
/// Limitations:
/// - Fixed 4x upscale factor
/// - High VRAM requirements due to temporal processing
/// - Processing speed limited by number of frames per batch
/// - May introduce subtle artifacts at scene transitions
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Video U-Net with temporal attention + temporal VAE
/// - Input channels: 8 (4 latent noise + 4 downscaled low-res)
/// - Output channels: 4 (latent space)
/// - Base channels: 320, multipliers [1, 2, 4, 4]
/// - Cross-attention dimension: 1024
/// - Temporal attention layers: 2 per block
/// - Temporal VAE: 3-frame kernel, 1 temporal layer
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Default frames: 16 at 24 FPS
/// - Upscale factor: 4x
/// - Scheduler: DDIM for efficient video inference
///
/// Reference: Zhou et al., "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var upscaler = new UpscaleAVideoModel&lt;float&gt;();
///
/// // Upscale a video (video-to-video super-resolution)
/// var upscaledVideo = upscaler.VideoToVideo(
///     inputVideo: lowResVideo,
///     prompt: "high resolution, sharp, detailed video",
///     strength: 0.7,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate video from an image (animate + upscale)
/// var videoFromImage = upscaler.GenerateFromImage(
///     inputImage: referenceFrame,
///     numFrames: 16,
///     fps: 24,
///     numInferenceSteps: 25);
/// </code>
/// </example>
public class UpscaleAVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default output video width (1024, 4x upscaled from 256).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default output video height (576, 4x upscaled from 144).
    /// </summary>
    public const int DefaultHeight = 576;

    /// <summary>
    /// Number of latent channels (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension (1024).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Input channels for the Video U-Net (8 = 4 latent + 4 low-res conditioning).
    /// </summary>
    /// <remarks>
    /// The Video U-Net receives concatenated latent noise and downscaled low-resolution
    /// video frames as conditioning, doubling the standard 4 latent channels.
    /// </remarks>
    private const int INPUT_CHANNELS = 8;

    /// <summary>
    /// Base channel count for the Video U-Net (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// Number of attention heads in the Video U-Net (8).
    /// </summary>
    private const int NUM_HEADS = 8;

    /// <summary>
    /// Number of temporal attention layers per block (2).
    /// </summary>
    /// <remarks>
    /// Two temporal layers per block provide stronger inter-frame consistency
    /// compared to single-layer temporal attention in simpler video models.
    /// </remarks>
    private const int NUM_TEMPORAL_LAYERS = 2;

    /// <summary>
    /// Upscale factor (4x).
    /// </summary>
    private const int UPSCALE_FACTOR = 4;

    /// <summary>
    /// Default number of frames per batch (16).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 16;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    #endregion

    #region Fields

    /// <summary>
    /// The Video U-Net noise predictor with temporal attention layers.
    /// </summary>
    private VideoUNetPredictor<T> _videoUNet;

    /// <summary>
    /// The temporal VAE for temporally coherent video encoding/decoding.
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// Optional conditioning module for guided video super-resolution.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

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
    public override bool SupportsImageToVideo => false;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => false;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => true;

    /// <inheritdoc />
    public override int ParameterCount =>
        _videoUNet.GetParameters().Length + _temporalVAE.GetParameters().Length;

    /// <summary>
    /// Gets the video upscale factor (4x).
    /// </summary>
    public int UpscaleFactor => UPSCALE_FACTOR;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of UpscaleAVideoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Upscale-A-Video defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDIM with SD settings for efficient video inference.
    /// </param>
    /// <param name="videoUNet">Custom Video U-Net. If null, creates the standard temporal U-Net.</param>
    /// <param name="temporalVAE">Custom Temporal VAE. If null, creates the standard temporal VAE.</param>
    /// <param name="conditioner">Optional conditioning module for guided upscaling.</param>
    /// <param name="defaultNumFrames">Default number of frames per batch (default: 16).</param>
    /// <param name="defaultFPS">Default frames per second (default: 24).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public UpscaleAVideoModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
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
        _conditioner = conditioner;

        InitializeLayers(videoUNet, temporalVAE, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Video U-Net and Temporal VAE using custom or default configurations.
    /// </summary>
    /// <param name="videoUNet">Custom Video U-Net, or null for defaults.</param>
    /// <param name="temporalVAE">Custom Temporal VAE, or null for defaults.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default Video U-Net:
    /// - Input: 8 channels (4 latent + 4 low-res conditioning)
    /// - Base channels: 320, multipliers [1, 2, 4, 4]
    /// - 2 temporal attention layers per block
    /// - Supports image conditioning for low-res input
    ///
    /// Default Temporal VAE:
    /// - 3-frame temporal kernel for inter-frame coherence
    /// - Non-causal mode for bidirectional temporal processing
    /// - 0.18215 latent scale factor
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_videoUNet), nameof(_temporalVAE))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? videoUNet,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: INPUT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM,
            numHeads: NUM_HEADS,
            numTemporalLayers: NUM_TEMPORAL_LAYERS,
            supportsImageConditioning: true);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 1,
            temporalKernelSize: 3,
            causalMode: false,
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
        return new UpscaleAVideoModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
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
            Name = "Upscale-A-Video",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Upscale-A-Video temporally consistent video super-resolution with diffusion",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "temporal-sr-diffusion");
        metadata.SetProperty("backbone", "Video-UNet-320B");
        metadata.SetProperty("upscale_factor", UPSCALE_FACTOR);
        metadata.SetProperty("input_channels", INPUT_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("temporal_layers", NUM_TEMPORAL_LAYERS);
        metadata.SetProperty("temporal_consistency", true);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
