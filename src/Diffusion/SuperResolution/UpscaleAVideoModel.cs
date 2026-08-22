using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.SuperResolution;

/// <summary>
/// Upscale-A-Video model for temporally consistent video super-resolution with diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Upscale-A-Video extends image super-resolution to video with temporal consistency,
/// using convolutional temporal modules and flow-guided recurrent propagation to achieve
/// flicker-free 4x upscaling of video content.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Video U-Net with released temporal 3D/spatial ResNet modules</description></item>
/// <item><description>7 input channels (4 latent + 3-channel noised low-resolution conditioning)</description></item>
/// <item><description>9 stage-level convolutional temporal modules plus zero-initialized Transformer3D temporal attention</description></item>
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
/// 2. The noised RGB low-resolution video is concatenated with the four-channel latent (7 input channels)
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
/// - Architecture: Video U-Net with temporal 3D CNN modules + temporal VAE
/// - Input channels: 7 (4 latent noise + 3-channel noised low-resolution video)
/// - Output channels: 4 (latent space)
/// - Base channels: 256, multipliers [1, 2, 2, 4]
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
///     prompt: "best quality, extremely detailed",
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
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.SuperResolution)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution", "https://arxiv.org/abs/2312.06640", Year = 2024, Authors = "Zhou et al.")]
public partial class UpscaleAVideoModel<T> : VideoDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        EnsureInitialized();
        RegisterParameterComponent("denoiser/video-unet", _videoUNet);
        RegisterParameterComponent("vae/temporal", _temporalVAE);
    }

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
    /// Input channels for the noisy target latent stream (4).
    /// </summary>
    /// <remarks>
    /// The predictor concatenates the distinct three-channel low-resolution video,
    /// producing the official seven-channel U-Net input (4 latent + 3 low resolution).
    /// </remarks>
    private const int INPUT_CHANNELS = LATENT_CHANNELS;

    /// <summary>
    /// Base channel count for the Video U-Net (256).
    /// </summary>
    private const int BASE_CHANNELS = 256;

    /// <summary>Channels in the low-resolution conditioning video.</summary>
    private const int CONDITION_CHANNELS = 3;

    /// <summary>Number of learned degradation/noise-level embeddings.</summary>
    private const int NUM_NOISE_LEVEL_EMBEDDINGS = 1000;

    /// <summary>
    /// Number of attention heads in the Video U-Net (8).
    /// </summary>
    private const int NUM_HEADS = 8;

    /// <summary>
    /// Upscale factor (4x).
    /// </summary>
    private const int UPSCALE_FACTOR = 4;

    /// <summary>
    /// Default temporal window from the released inference pipeline (8 frames).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 8;

    /// <summary>
    /// Default frames per second (24).
    /// </summary>
    private const int DEFAULT_FPS = 24;

    /// <summary>Temporal denoising window used by the reference implementation.</summary>
    public const int ReferenceWindowSize = 8;

    /// <summary>Overlap between adjacent temporal denoising windows.</summary>
    public const int ReferenceWindowOverlap = 2;

    /// <summary>Stable Diffusion x4-upscaler VAE scaling factor.</summary>
    public const double ReferenceLatentScaleFactor = 0.08333;

    /// <summary>Maximum low-resolution noise level used by the reference pipeline.</summary>
    public const int ReferenceMaximumNoiseLevel = 350;

    #endregion

    #region Fields

    /// <summary>
    /// The Video U-Net noise predictor with released convolutional temporal modules.
    /// </summary>
    private VideoUNetPredictor<T>? _videoUNet;

    /// <summary>
    /// The temporal VAE for temporally coherent video encoding/decoding.
    /// </summary>
    private TemporalVAE<T>? _temporalVAE;


    /// <summary>
    /// Optional conditioning module for guided video super-resolution.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;
    // The SD-x4 conditioner is frozen while Upscale-A-Video optimizes the denoiser. Keep the two
    // classifier-free-guidance prompt embeddings in arena-pinned storage across repeated requests.
    // Explicit positive/negative slots bound memory even when prompt text changes.
    private readonly object _conditioningCacheLock = new();
    private string? _cachedPrompt;
    private Tensor<T>? _cachedPromptConditioning;
    private string? _cachedNegativePrompt;
    private Tensor<T>? _cachedNegativeConditioning;
    // Seed for the deferred (lazy) init path: the constructor only eager-inits when an explicit
    // predictor/VAE is passed, so without capturing the seed the lazy EnsureInitialized() built the
    // sub-models with a null seed — dropping the requested seed and making construction non-reproducible.
    private readonly int? _seed;

    // Conditional training state is scoped to a single TrainConditioned call. The base diffusion
    // trainer owns timestep sampling, target-noise construction, autodiff, and optimization; these
    // fields provide the paper-specific low-resolution RGB and text context to its virtual hooks.
    private readonly object _trainingContextLock = new();
    private Tensor<T>? _trainingVideoCondition;
    private Tensor<T>? _trainingTextConditioning;
    private int _trainingNoiseLevel;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor { get { EnsureInitialized(); return _videoUNet; } }

    /// <inheritdoc />
    public override IVAEModel<T> VAE { get { EnsureInitialized(); return _temporalVAE; } }

    /// <inheritdoc />
    public override IVAEModel<T>? TemporalVAE { get { EnsureInitialized(); return _temporalVAE; } }

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


    /// <summary>
    /// Gets the video upscale factor (4x).
    /// </summary>
    public int UpscaleFactor => UPSCALE_FACTOR;

    /// <summary>
    /// The released Stable Diffusion first-stage VAE, CLIP encoder, and pretrained spatial U-Net
    /// layers are frozen during temporal denoiser fine-tuning. Keep them in checkpoint
    /// serialization, but do not allocate gradients or optimizer moments for them.
    /// </summary>
    protected override object TrainingParameterRoot
    {
        get
        {
            EnsureInitialized();
            return _videoUNet.TemporalTrainingLayers;
        }
    }

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
            options is null
                ? new DiffusionModelOptions<T>
                {
                    TrainTimesteps = 1000,
                    BetaStart = 0.00085,
                    BetaEnd = 0.012,
                    BetaSchedule = BetaSchedule.ScaledLinear
                }
                : new DiffusionModelOptions<T>(options),
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;
        _seed = seed;

        if (videoUNet is not null || temporalVAE is not null)
            InitializeLayers(videoUNet, temporalVAE, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_videoUNet), nameof(_temporalVAE))]
    private void EnsureInitialized()
    {
        if (_videoUNet is null || _temporalVAE is null)
            InitializeLayers(null, null, _seed);
    }


    /// <summary>
    /// Initializes the Video U-Net and Temporal VAE using custom or default configurations.
    /// </summary>
    /// <param name="videoUNet">Custom Video U-Net, or null for defaults.</param>
    /// <param name="temporalVAE">Custom Temporal VAE, or null for defaults.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default Video U-Net:
    /// - Input: 7 channels (4 latent + 3-channel noised low-resolution conditioning)
    /// - Base channels: 256, multipliers [1, 2, 2, 4]
    /// - 9 stage-level convolutional temporal modules plus Transformer3D temporal attention
    /// - Supports image conditioning for low-res input
    ///
    /// Default Temporal VAE:
    /// - 3-frame temporal kernel for inter-frame coherence
    /// - Non-causal mode for bidirectional temporal processing
    /// - 0.08333 Stable Diffusion x4-upscaler latent scale factor
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
            channelMultipliers: new[] { 1, 2, 2, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 1, 2, 3 },
            contextDim: CROSS_ATTENTION_DIM,
            numHeads: NUM_HEADS,
            numTemporalLayers: 1,
            supportsImageConditioning: true,
            inputHeight: 128,
            inputWidth: 128,
            numFrames: ReferenceWindowSize,
            seed: seed,
            imageConditionChannels: CONDITION_CHANNELS,
            concatenateImageCondition: true,
            numClassEmbeddings: NUM_NOISE_LEVEL_EMBEDDINGS,
            architectureProfile: VideoUNetArchitectureProfile.UpscaleAVideo);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4 },
            numTemporalLayers: 1,
            temporalKernelSize: 3,
            causalMode: false,
            latentScaleFactor: ReferenceLatentScaleFactor,
            seed: seed);
    }

    #endregion

    #region Training

    /// <summary>
    /// Trains the Upscale-A-Video denoiser on a low/high-resolution video pair.
    /// </summary>
    /// <param name="lowResolutionVideo">Low-resolution conditioning video in [C,H,W], [F,C,H,W], or [B,F,C,H,W] layout.</param>
    /// <param name="highResolutionVideo">Four-times-larger clean target in the corresponding layout.</param>
    /// <param name="prompt">Text condition associated with the training clip.</param>
    /// <param name="noiseLevel">SD x4 degradation-noise class label in [0,350].</param>
    /// <remarks>
    /// Upscale-A-Video does not train the denoiser against the low-resolution RGB tensor. The
    /// clean target is the temporal-VAE latent of the high-resolution video. Scheduler noise is
    /// added to that four-channel latent, while separately noised low-resolution RGB is supplied
    /// as the three-channel condition, producing the released seven-channel U-Net input.
    /// </remarks>
    public void TrainConditioned(
        Tensor<T> lowResolutionVideo,
        Tensor<T> highResolutionVideo,
        string prompt,
        int noiseLevel = 20)
    {
        if (lowResolutionVideo is null)
            throw new ArgumentNullException(nameof(lowResolutionVideo));
        if (highResolutionVideo is null)
            throw new ArgumentNullException(nameof(highResolutionVideo));
        if (prompt is null)
            throw new ArgumentNullException(nameof(prompt));
        if ((uint)noiseLevel > ReferenceMaximumNoiseLevel)
            throw new ArgumentOutOfRangeException(nameof(noiseLevel), noiseLevel,
                $"Noise level must be in [0, {ReferenceMaximumNoiseLevel}].");

        lock (_trainingContextLock)
        {
            EnsureInitialized();
            if (_conditioner is null)
                throw new InvalidOperationException(
                    "Upscale-A-Video training requires the Stable Diffusion x4 text conditioner.");

            var low = NormalizeVideoBatch(lowResolutionVideo, nameof(lowResolutionVideo));
            var high = NormalizeVideoBatch(highResolutionVideo, nameof(highResolutionVideo));
            ValidateTrainingPair(low, high);

            var cleanCondition = Engine.TensorPermute(low, [0, 2, 1, 3, 4]).Contiguous();
            var conditionNoise = DiffusionNoiseHelper<T>.SampleGaussian(
                cleanCondition.Shape.ToArray(), RandomGenerator);
            _trainingVideoCondition = new Tensor<T>(
                cleanCondition._shape,
                Scheduler.AddNoise(cleanCondition.ToVector(), conditionNoise.ToVector(), noiseLevel));
            _trainingTextConditioning = GetCachedTextConditioning(prompt, negativeSlot: false);
            _trainingNoiseLevel = noiseLevel;

            try
            {
                base.Train(low, high);
            }
            finally
            {
                _trainingVideoCondition?.Dispose();
                _trainingTextConditioning?.Dispose();
                _trainingVideoCondition = null;
                _trainingTextConditioning = null;
                _trainingNoiseLevel = 0;
            }
        }
    }

    protected override Tensor<T> PrepareTrainingSample(
        Tensor<T> input,
        Tensor<T> expectedOutput)
    {
        EnsureInitialized();
        var target = NormalizeVideoBatch(expectedOutput, nameof(expectedOutput));
        // TensorPermute returns a strided view. TemporalVAE's frame extraction uses raw spans,
        // so materialize the layout at this explicit component boundary.
        var targetNCFHW = Engine.TensorPermute(target, [0, 2, 1, 3, 4]).Contiguous();
        // The VAE is the frozen Stable Diffusion first stage. Sampling from its posterior is the
        // reference latent-diffusion training contract; gradients are recorded only after this
        // preparation hook returns, so optimization remains focused on the denoiser.
        return _temporalVAE.EncodeVideoForDiffusion(targetNCFHW, sampleMode: true);
    }

    protected override Tensor<T> PredictTrainingNoise(
        Tensor<T> noisySample,
        int[] timesteps,
        bool isBatched,
        Tensor<T> input,
        Tensor<T> expectedOutput)
    {
        EnsureInitialized();
        var condition = _trainingVideoCondition ?? throw new InvalidOperationException(
            $"Use {nameof(TrainConditioned)} so the low-resolution video condition is available.");
        var textConditioning = _trainingTextConditioning ?? throw new InvalidOperationException(
            $"Use {nameof(TrainConditioned)} so the text condition is available.");

        int batchSize = noisySample.Shape[0];
        if (!isBatched || batchSize == 1)
            return _videoUNet.PredictNoiseWithVideoCondition(
                noisySample, timesteps[0], condition, textConditioning, _trainingNoiseLevel);

        if (timesteps.Length != batchSize)
            throw new ArgumentException(
                $"Expected {batchSize} training timesteps, got {timesteps.Length}.",
                nameof(timesteps));

        // A distinct timestep per batch element is the canonical DDPM training contract. The
        // Video U-Net accepts one scalar timestep, so retain tensor rank and concatenate the
        // independently conditioned predictions along the batch axis.
        var predictions = new Tensor<T>[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            var sample = Engine.TensorSlice(
                noisySample,
                [b, 0, 0, 0, 0],
                [1, noisySample.Shape[1], noisySample.Shape[2], noisySample.Shape[3], noisySample.Shape[4]]);
            var sampleCondition = Engine.TensorSlice(
                condition,
                [b, 0, 0, 0, 0],
                [1, condition.Shape[1], condition.Shape[2], condition.Shape[3], condition.Shape[4]]);
            predictions[b] = _videoUNet.PredictNoiseWithVideoCondition(
                sample, timesteps[b], sampleCondition, textConditioning, _trainingNoiseLevel);
        }

        return Engine.TensorConcatenate(predictions, axis: 0);
    }

    private Tensor<T> NormalizeVideoBatch(Tensor<T> video, string parameterName)
    {
        return video.Rank switch
        {
            3 => Engine.Reshape(video,
                [1, 1, video.Shape[0], video.Shape[1], video.Shape[2]]),
            4 => Engine.Reshape(video,
                [1, video.Shape[0], video.Shape[1], video.Shape[2], video.Shape[3]]),
            5 => video,
            _ => throw new ArgumentException(
                "Upscale-A-Video expects [C,H,W], [F,C,H,W], or [B,F,C,H,W] video tensors.",
                parameterName)
        };
    }

    private static void ValidateTrainingPair(Tensor<T> low, Tensor<T> high)
    {
        if (low.Shape[0] != high.Shape[0] || low.Shape[1] != high.Shape[1])
            throw new ArgumentException(
                "Low- and high-resolution videos must have matching batch and frame counts.",
                nameof(high));
        if (low.Shape[2] != CONDITION_CHANNELS || high.Shape[2] != CONDITION_CHANNELS)
            throw new ArgumentException("Upscale-A-Video training requires three-channel RGB videos.");
        if (high.Shape[3] != low.Shape[3] * UPSCALE_FACTOR ||
            high.Shape[4] != low.Shape[4] * UPSCALE_FACTOR)
            throw new ArgumentException(
                $"The high-resolution target must be {UPSCALE_FACTOR}x the low-resolution video.",
                nameof(high));
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Runs the Upscale-A-Video super-resolution path with a distinct low-resolution
    /// conditioning latent for every frame.
    /// </summary>
    /// <param name="lowResolutionVideo">Input video in [B,F,C,H,W] layout.</param>
    /// <param name="prompt">Optional text guidance prompt.</param>
    /// <param name="numInferenceSteps">DDIM denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional deterministic inference seed.</param>
    /// <param name="noiseLevel">Stable Diffusion x4 degradation/noise class in [0, 350].</param>
    /// <param name="temporalWindowSize">Number of frames processed per temporal window.</param>
    /// <param name="temporalWindowOverlap">Number of frames shared by adjacent windows.</param>
    /// <param name="forwardFlows">Optional forward optical flows; must be supplied with <paramref name="backwardFlows"/>.</param>
    /// <param name="backwardFlows">Optional backward optical flows; must be supplied with <paramref name="forwardFlows"/>.</param>
    /// <param name="propagationSteps">Denoising-step indexes at which bidirectional flow propagation is applied.</param>
    /// <param name="negativePrompt">Negative text used by classifier-free guidance.</param>
    /// <returns>Four-times-upscaled video in [B,F,C,4H,4W] layout.</returns>
    public Tensor<T> Upscale(
        Tensor<T> lowResolutionVideo,
        string prompt = "best quality, extremely detailed",
        int numInferenceSteps = 75,
        double guidanceScale = 9.0,
        int? seed = null,
        int noiseLevel = 20,
        int temporalWindowSize = ReferenceWindowSize,
        int temporalWindowOverlap = ReferenceWindowOverlap,
        Tensor<T>? forwardFlows = null,
        Tensor<T>? backwardFlows = null,
        IReadOnlyCollection<int>? propagationSteps = null,
        string negativePrompt = "blur, worst quality")
    {
        EnsureInitialized();
        int originalRank = lowResolutionVideo.Rank;
        if (originalRank == 3)
            lowResolutionVideo = Engine.Reshape(lowResolutionVideo,
                [1, 1, lowResolutionVideo.Shape[0], lowResolutionVideo.Shape[1], lowResolutionVideo.Shape[2]]);
        else if (originalRank == 4)
            lowResolutionVideo = Engine.Reshape(lowResolutionVideo,
                [1, lowResolutionVideo.Shape[0], lowResolutionVideo.Shape[1], lowResolutionVideo.Shape[2], lowResolutionVideo.Shape[3]]);
        else if (originalRank != 5)
            throw new ArgumentException("Upscale-A-Video expects [C,H,W], [F,C,H,W], or [B,F,C,H,W] input.", nameof(lowResolutionVideo));
        if (numInferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numInferenceSteps));
        if ((uint)noiseLevel > ReferenceMaximumNoiseLevel)
            throw new ArgumentOutOfRangeException(nameof(noiseLevel), noiseLevel,
                $"Noise level must be in [0, {ReferenceMaximumNoiseLevel}].");
        if (temporalWindowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(temporalWindowSize));
        if (temporalWindowOverlap < 0 || temporalWindowOverlap >= temporalWindowSize)
            throw new ArgumentOutOfRangeException(nameof(temporalWindowOverlap));
        if ((forwardFlows is null) != (backwardFlows is null))
            throw new ArgumentException("Forward and backward optical flows must be supplied together.");

        int batch = lowResolutionVideo.Shape[0];
        int frames = lowResolutionVideo.Shape[1];
        int channels = lowResolutionVideo.Shape[2];
        int height = lowResolutionVideo.Shape[3];
        int width = lowResolutionVideo.Shape[4];

        if (channels != CONDITION_CHANNELS)
            throw new ArgumentException(
                $"Upscale-A-Video expects {CONDITION_CHANNELS}-channel RGB input, got {channels}.",
                nameof(lowResolutionVideo));

        // The official SD-x4 contract does not VAE-encode the low-resolution input.
        // It adds scheduler noise at the selected degradation level to RGB directly,
        // then concatenates those 3 channels with the 4-channel target latent.
        // TensorPermute returns a strided view and the scheduler consumes raw vectors, so
        // materialize the NCFHW component boundary just as TrainConditioned does.
        var cleanCondition = Engine.TensorPermute(
            lowResolutionVideo, [0, 2, 1, 3, 4]).Contiguous();

        if (_conditioner is null)
            throw new InvalidOperationException(
                "Upscale-A-Video requires the Stable Diffusion x4 Upscaler CLIP text " +
                "encoder (1024-dimensional, 23-layer) for its Transformer3D cross-attention. " +
                "Pass that released conditioner; reducing guidance does not remove the model's " +
                "text-conditioning layers.");
        if (_conditioner.EmbeddingDimension != CROSS_ATTENTION_DIM)
            throw new InvalidOperationException(
                $"Upscale-A-Video requires {CROSS_ATTENTION_DIM}-dimensional text conditioning, " +
                $"but the supplied conditioner produces {_conditioner.EmbeddingDimension} dimensions.");
        bool useGuidance = guidanceScale > 1.0 && _videoUNet.SupportsCFG;
        using var textConditioning = GetCachedTextConditioning(prompt, negativeSlot: false);
        using Tensor<T>? unconditional = useGuidance
            ? GetCachedTextConditioning(negativePrompt, negativeSlot: true)
            : null;

        Scheduler.SetTimesteps(numInferenceSteps);
        var rng = CreateInferenceRng(seed);
        var conditionNoise = DiffusionNoiseHelper<T>.SampleGaussian(cleanCondition.Shape.ToArray(), rng);
        var noisedCondition = new Tensor<T>(cleanCondition._shape,
            Scheduler.AddNoise(cleanCondition.ToVector(), conditionNoise.ToVector(), noiseLevel));
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(
            [batch, LATENT_CHANNELS, frames, height, width], rng);
        var propagationSet = propagationSteps is null
            ? null
            : new HashSet<int>(propagationSteps);

        for (int stepIndex = 0; stepIndex < Scheduler.Timesteps.Length; stepIndex++)
        {
            int timestep = Scheduler.Timesteps[stepIndex];
            var conditionalPrediction = PredictWindowedNoise(
                latents, timestep, noisedCondition, textConditioning, noiseLevel,
                temporalWindowSize, temporalWindowOverlap);
            Tensor<T> prediction = conditionalPrediction;
            if (useGuidance && unconditional is not null)
            {
                var unconditionalPrediction = PredictWindowedNoise(
                    latents, timestep, noisedCondition, unconditional, noiseLevel,
                    temporalWindowSize, temporalWindowOverlap);
                prediction = ApplyGuidanceVideo(
                    unconditionalPrediction, conditionalPrediction, guidanceScale);
            }

            bool propagate = forwardFlows is not null && backwardFlows is not null &&
                propagationSet?.Contains(stepIndex) == true;
            if (propagate)
            {
                var x0 = PredictOriginalSample(latents, prediction, timestep);
                x0 = PropagateBidirectionally(x0, forwardFlows!, backwardFlows!);
                latents = StepFromOriginalSample(x0, prediction, stepIndex);
            }
            else
            {
                latents = SchedulerStepVideo(latents, prediction, timestep);
            }
        }

        var decodedNCFHW = _temporalVAE.DecodeVideoFromDiffusion(latents);
        var decoded = Engine.TensorPermute(
            decodedNCFHW, [0, 2, 1, 3, 4]).Contiguous();
        return originalRank switch
        {
            3 => Engine.Reshape(decoded, [decoded.Shape[2], decoded.Shape[3], decoded.Shape[4]]),
            4 => Engine.Reshape(decoded, [decoded.Shape[1], decoded.Shape[2], decoded.Shape[3], decoded.Shape[4]]),
            _ => decoded
        };
    }

    private Tensor<T> GetCachedTextConditioning(string prompt, bool negativeSlot)
    {
        if (_conditioner is null)
            throw new InvalidOperationException(
                "Upscale-A-Video requires a text conditioner before encoding prompts.");

        lock (_conditioningCacheLock)
        {
            string? cachedKey = negativeSlot ? _cachedNegativePrompt : _cachedPrompt;
            Tensor<T>? cachedValue = negativeSlot
                ? _cachedNegativeConditioning
                : _cachedPromptConditioning;
            if (cachedValue is not null && string.Equals(cachedKey, prompt, StringComparison.Ordinal))
                return DetachConditioning(cachedValue);

            using var tokens = _conditioner.Tokenize(prompt);
            using var encoded = _conditioner.EncodeText(tokens);
            // Cache ordinary owned storage. Replacing it cannot return memory to a pool while an
            // in-flight request is reading its detached result, and the old entry becomes safely
            // collectible after the lock releases.
            var persistent = new Tensor<T>(
                encoded.AsSpan().ToArray(), encoded.Shape.ToArray());

            if (negativeSlot)
            {
                _cachedNegativePrompt = prompt;
                _cachedNegativeConditioning = persistent;
            }
            else
            {
                _cachedPrompt = prompt;
                _cachedPromptConditioning = persistent;
            }

            return DetachConditioning(persistent);
        }
    }

    private static Tensor<T> DetachConditioning(Tensor<T> source)
        => new(source.AsSpan().ToArray(), source.Shape.ToArray());

    /// <summary>
    /// Runs the temporal U-Net in the reference eight-frame windows and blends every
    /// repeated frame 50/50 in encounter order, matching the released pipeline.
    /// </summary>
    private Tensor<T> PredictWindowedNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> condition,
        Tensor<T>? textConditioning,
        int noiseLevel,
        int windowSize,
        int overlap)
    {
        var predictor = _videoUNet ?? throw new InvalidOperationException("Video U-Net is not initialized.");
        int frames = latents.Shape[2];
        if (frames <= windowSize)
            return predictor.PredictNoiseWithVideoCondition(
                latents, timestep, condition, textConditioning, noiseLevel);

        int batch = latents.Shape[0];
        int latentChannels = latents.Shape[1];
        int conditionChannels = condition.Shape[1];
        int height = latents.Shape[3];
        int width = latents.Shape[4];
        int stride = windowSize - overlap;
        var predictions = new Tensor<T>?[frames];
        int previousStart = -1;

        for (int candidate = 0; candidate < frames; candidate += stride)
        {
            int end = System.Math.Min(frames, candidate + windowSize);
            int start = end - candidate < windowSize ? end - windowSize : candidate;
            if (start == previousStart)
                continue;
            previousStart = start;

            var latentWindow = Engine.TensorSlice(latents,
                [0, 0, start, 0, 0], [batch, latentChannels, windowSize, height, width]);
            var conditionWindow = Engine.TensorSlice(condition,
                [0, 0, start, 0, 0], [batch, conditionChannels, windowSize, height, width]);
            var windowPrediction = predictor.PredictNoiseWithVideoCondition(
                latentWindow, timestep, conditionWindow, textConditioning, noiseLevel);

            for (int offset = 0; offset < windowSize; offset++)
            {
                int frame = start + offset;
                var framePrediction = Engine.TensorSlice(windowPrediction,
                    [0, 0, offset, 0, 0], [batch, latentChannels, 1, height, width]);
                predictions[frame] = predictions[frame] is null
                    ? framePrediction
                    : Engine.TensorMultiplyScalar(
                        Engine.TensorAdd(predictions[frame]!, framePrediction),
                        NumOps.FromDouble(0.5));
            }
        }

        if (predictions.Any(static p => p is null))
            throw new InvalidOperationException("Temporal window schedule did not cover every frame.");
        return Engine.TensorConcatenate(predictions.Select(static p => p!).ToArray(), axis: 2);
    }

    private Tensor<T> PredictOriginalSample(
        Tensor<T> latents,
        Tensor<T> noisePrediction,
        int timestep)
    {
        double alpha = NumOps.ToDouble(Scheduler.GetAlphaCumulativeProduct(timestep));
        T sqrtAlpha = NumOps.FromDouble(System.Math.Sqrt(alpha));
        T sqrtOneMinusAlpha = NumOps.FromDouble(System.Math.Sqrt(System.Math.Max(0.0, 1.0 - alpha)));
        return Engine.TensorDivideScalar(
            Engine.TensorSubtract(
                latents,
                Engine.TensorMultiplyScalar(noisePrediction, sqrtOneMinusAlpha)),
            sqrtAlpha);
    }

    /// <summary>
    /// Reconstructs x(t-1) from a possibly propagated x0 and the original epsilon,
    /// which is the released scheduler's split step_v0 / step_vt contract at eta=0.
    /// </summary>
    private Tensor<T> StepFromOriginalSample(
        Tensor<T> originalSample,
        Tensor<T> noisePrediction,
        int stepIndex)
    {
        int previousTimestep = stepIndex + 1 < Scheduler.Timesteps.Length
            ? Scheduler.Timesteps[stepIndex + 1]
            : 0;
        double alphaPrevious = NumOps.ToDouble(
            Scheduler.GetAlphaCumulativeProduct(previousTimestep));
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(originalSample,
                NumOps.FromDouble(System.Math.Sqrt(alphaPrevious))),
            Engine.TensorMultiplyScalar(noisePrediction,
                NumOps.FromDouble(System.Math.Sqrt(System.Math.Max(0.0, 1.0 - alphaPrevious)))));
    }

    /// <summary>
    /// Applies the released non-learned propagation module: backward pass followed by
    /// forward pass, nearest flow warping, 50/50 fusion, and forward/backward consistency
    /// rejection with alpha1=0.001 and alpha2=0.05.
    /// </summary>
    private Tensor<T> PropagateBidirectionally(
        Tensor<T> features,
        Tensor<T> forwardFlows,
        Tensor<T> backwardFlows)
    {
        ValidatePropagationFlows(features, forwardFlows, backwardFlows);
        var backward = PropagateDirection(
            features, forwardFlows, backwardFlows, reverse: true);
        return PropagateDirection(
            backward, backwardFlows, forwardFlows, reverse: false);
    }

    private static void ValidatePropagationFlows(
        Tensor<T> features,
        Tensor<T> forwardFlows,
        Tensor<T> backwardFlows)
    {
        if (forwardFlows.Rank != 5 || backwardFlows.Rank != 5)
            throw new ArgumentException("Optical flows must use [B,2,F-1,H,W] layout.");
        int expectedFrames = features.Shape[2] - 1;
        int[] expected = [features.Shape[0], 2, expectedFrames, features.Shape[3], features.Shape[4]];
        if (!forwardFlows.Shape.ToArray().SequenceEqual(expected) ||
            !backwardFlows.Shape.ToArray().SequenceEqual(expected))
            throw new ArgumentException(
                $"Optical flow shape must be [{string.Join(",", expected)}] for these latents.");
    }

    private Tensor<T> PropagateDirection(
        Tensor<T> input,
        Tensor<T> propagationFlows,
        Tensor<T> consistencyFlows,
        bool reverse)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int frames = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];
        var output = new Tensor<T>?[frames];
        Tensor<T>? propagated = null;

        for (int iteration = 0; iteration < frames; iteration++)
        {
            int frameIndex = reverse ? frames - 1 - iteration : iteration;
            var current5D = Engine.TensorSlice(input,
                [0, 0, frameIndex, 0, 0], [batch, channels, 1, height, width]);
            var current = Engine.Reshape(current5D, [batch, channels, height, width]);
            if (propagated is null)
            {
                propagated = current;
            }
            else
            {
                int flowIndex = reverse ? frameIndex : frameIndex - 1;
                var flow = ExtractFlowFrame(propagationFlows, flowIndex);
                var reverseFlow = ExtractFlowFrame(consistencyFlows, flowIndex);
                var warped = WarpNearest(propagated, flow);
                var mask = CreateConsistencyMask(flow, reverseFlow, channels);
                var fused = Engine.TensorMultiplyScalar(
                    Engine.TensorAdd(warped, current), NumOps.FromDouble(0.5));
                var inverseMask = Engine.TensorSubtract(
                    Engine.TensorAddScalar(
                        Engine.TensorMultiplyScalar(mask, NumOps.Zero), NumOps.One),
                    mask);
                propagated = Engine.TensorAdd(
                    Engine.TensorMultiply(mask, fused),
                    Engine.TensorMultiply(inverseMask, current));
            }

            output[frameIndex] = Engine.Reshape(propagated,
                [batch, channels, 1, height, width]);
        }

        return Engine.TensorConcatenate(output.Select(static x => x!).ToArray(), axis: 2);
    }

    private Tensor<T> ExtractFlowFrame(Tensor<T> flows, int frameIndex)
    {
        var sliced = Engine.TensorSlice(flows,
            [0, 0, frameIndex, 0, 0],
            [flows.Shape[0], 2, 1, flows.Shape[3], flows.Shape[4]]);
        return Engine.Reshape(sliced, [flows.Shape[0], 2, flows.Shape[3], flows.Shape[4]]);
    }

    internal Tensor<T> WarpNearest(Tensor<T> input, Tensor<T> flow)
    {
        int batch = input.Shape[0];
        int height = input.Shape[2];
        int width = input.Shape[3];
        var identity = new Tensor<T>([batch, 2, 3]);
        for (int b = 0; b < batch; b++)
        {
            identity[b, 0, 0] = NumOps.One;
            identity[b, 1, 1] = NumOps.One;
        }

        var baseGrid = Engine.AffineGrid(identity, height, width);
        var scale = new Tensor<T>([1, 1, 1, 2]);
        scale[0, 0, 0, 0] = NumOps.FromDouble(width <= 1 ? 0.0 : 2.0 / (width - 1));
        scale[0, 0, 0, 1] = NumOps.FromDouble(height <= 1 ? 0.0 : 2.0 / (height - 1));
        var normalizedFlow = Engine.TensorMultiply(
            Engine.TensorPermute(flow, [0, 2, 3, 1]), scale);
        var grid = Engine.TensorAdd(baseGrid, normalizedFlow);
        return Engine.GridSample(
            input, grid, GridSampleMode.Nearest, GridSamplePadding.Zeros, alignCorners: true);
    }

    private Tensor<T> CreateConsistencyMask(
        Tensor<T> flow,
        Tensor<T> reverseFlow,
        int outputChannels)
    {
        int height = flow.Shape[2];
        int width = flow.Shape[3];
        var warpedReverse = WarpNearest(reverseFlow, flow);
        var consistencyError = Engine.ReduceSum(
            Engine.TensorSquare(Engine.TensorAdd(flow, warpedReverse)), [1], keepDims: true);
        var flowEnergy = Engine.ReduceSum(
            Engine.TensorAdd(Engine.TensorSquare(flow), Engine.TensorSquare(warpedReverse)),
            [1], keepDims: true);
        var threshold = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(flowEnergy, NumOps.FromDouble(0.001)),
            NumOps.FromDouble(0.05));
        var singleChannelMask = Engine.TensorLessThan(consistencyError, threshold);
        return Engine.TensorBroadcastTo(
            singleChannelMask, [flow.Shape[0], outputChannels, height, width]);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        EnsureInitialized();
        return _videoUNet.PredictNoiseWithImageCondition(
            latents, timestep, imageEmbedding, textConditioning: null);
    }

    #endregion

    #region IParameterizable Implementation



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
        EnsureInitialized();
        return new UpscaleAVideoModel<T>(
            architecture: Architecture,
            options: new DiffusionModelOptions<T>((DiffusionModelOptions<T>)GetOptions()),
            scheduler: CloneScheduler(Scheduler),
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS,
            seed: _seed);
    }

    private static INoiseScheduler<T> CloneScheduler(INoiseScheduler<T> scheduler)
    {
        object? created = Activator.CreateInstance(scheduler.GetType(), scheduler.Config);
        if (created is not INoiseScheduler<T> clone)
            throw new InvalidOperationException(
                $"Scheduler {scheduler.GetType().Name} must expose a constructor accepting SchedulerConfig<{typeof(T).Name}> to support model cloning.");

        var state = scheduler.GetState().ToDictionary(
            pair => pair.Key,
            pair => pair.Value is int[] values ? (object)values.ToArray() : pair.Value);
        clone.LoadState(state);
        return clone;
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        EnsureInitialized();
        var metadata = new ModelMetadata<T>
        {
            Name = "Upscale-A-Video",
            Version = "1.0",
            Description = "Upscale-A-Video temporally consistent video super-resolution with diffusion",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "temporal-sr-diffusion");
        metadata.SetProperty("backbone", "Video-UNet-256 [1,2,2,4]");
        metadata.SetProperty("upscale_factor", UPSCALE_FACTOR);
        metadata.SetProperty("input_channels", INPUT_CHANNELS + CONDITION_CHANNELS);
        metadata.SetProperty("latent_input_channels", INPUT_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("temporal_modules", _videoUNet.TemporalModuleCount);
        metadata.SetProperty("condition_channels", CONDITION_CHANNELS);
        metadata.SetProperty("noise_level_embeddings", NUM_NOISE_LEVEL_EMBEDDINGS);
        metadata.SetProperty("temporal_window", ReferenceWindowSize);
        metadata.SetProperty("temporal_window_overlap", ReferenceWindowOverlap);
        metadata.SetProperty("temporal_consistency", true);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            lock (_conditioningCacheLock)
            {
                _cachedPromptConditioning?.Dispose();
                _cachedPromptConditioning = null;
                _cachedPrompt = null;
                _cachedNegativeConditioning?.Dispose();
                _cachedNegativeConditioning = null;
                _cachedNegativePrompt = null;
            }
        }

        base.Dispose(disposing);
    }

    #endregion
}
