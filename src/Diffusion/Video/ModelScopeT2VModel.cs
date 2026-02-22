using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// ModelScope Text-to-Video model with temporal U-Net for short video clip generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ModelScope T2V extends the Stable Diffusion U-Net architecture with temporal convolution
/// and temporal attention modules, enabling text-to-video generation. It was one of the first
/// open-source text-to-video models trained on the WebVid-10M dataset.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Video U-Net based on SD 1.5 with temporal extension blocks</description></item>
/// <item><description>320 base channels with [1, 2, 4, 4] channel multipliers</description></item>
/// <item><description>1 temporal attention layer per block for inter-frame consistency</description></item>
/// <item><description>CLIP text encoder for 1024-dim cross-attention conditioning</description></item>
/// <item><description>Standard SD VAE for per-frame spatial compression (4 latent channels)</description></item>
/// <item><description>DDPM noise schedule with scaled linear beta</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> ModelScope T2V generates short video clips from text prompts.
///
/// How ModelScope T2V works:
/// 1. Text prompt is encoded by CLIP into 1024-dim embeddings
/// 2. Each video frame is encoded by the SD VAE into 4 latent channels
/// 3. The temporal U-Net processes latent frames with temporal attention and convolution
/// 4. Temporal attention ensures frames are consistent with each other
/// 5. The VAE decodes each latent frame back to pixel space
///
/// Key characteristics:
/// - Based on SD 1.5 architecture with temporal blocks added
/// - Trained on WebVid-10M dataset (10M video-text pairs)
/// - 256x256 base resolution with cascaded upscaling to 512x512
/// - 16 frames at 8 FPS by default (~2 seconds)
/// - One of the first open-source text-to-video models
///
/// When to use ModelScope T2V:
/// - Simple text-to-video generation
/// - Research on temporal attention mechanisms
/// - Lightweight video generation on modest hardware
/// - Building on established SD 1.5 ecosystem
///
/// Limitations:
/// - Low resolution (256-512px)
/// - Short duration (16 frames)
/// - Quality below modern video models
/// - Single temporal attention layer limits motion coherence
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Video U-Net (SD 1.5 + temporal blocks)
/// - Base channels: 320, multipliers [1, 2, 4, 4]
/// - Attention heads: 8
/// - ResNet blocks per level: 2
/// - Temporal attention layers: 1 per block
/// - Latent channels: 4 (standard SD VAE)
/// - Cross-attention dimension: 1024 (CLIP)
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Default: 16 frames at 8 FPS (~2 seconds)
/// - Training dataset: WebVid-10M
///
/// Reference: Wang et al., "ModelScope Text-to-Video Technical Report", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var modelScope = new ModelScopeT2VModel&lt;float&gt;();
///
/// // Generate video from text
/// var video = modelScope.GenerateFromText(
///     prompt: "A teddy bear painting a portrait",
///     width: 256,
///     height: 256,
///     numFrames: 16,
///     fps: 8,
///     numInferenceSteps: 50,
///     guidanceScale: 9.0);
/// </code>
/// </example>
public class ModelScopeT2VModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the standard SD VAE (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension from the CLIP text encoder (1024).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Base channel count for the Video U-Net (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    /// <summary>
    /// Number of attention heads (8).
    /// </summary>
    private const int NUM_HEADS = 8;

    /// <summary>
    /// Number of temporal attention layers per block (1).
    /// </summary>
    private const int NUM_TEMPORAL_LAYERS = 1;

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
    /// The Video U-Net noise predictor with temporal attention.
    /// </summary>
    private VideoUNetPredictor<T> _videoUNet;

    /// <summary>
    /// The standard SD VAE for per-frame spatial compression.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The CLIP text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _videoUNet;

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
    public override int ParameterCount =>
        _videoUNet.GetParameters().Length + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of ModelScopeT2VModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses ModelScope defaults:
    /// scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses DDPM scheduler with SD settings.
    /// </param>
    /// <param name="videoUNet">Custom Video U-Net. If null, creates the standard temporal U-Net.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD VAE.</param>
    /// <param name="conditioner">CLIP text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 16).</param>
    /// <param name="defaultFPS">Default frames per second (default: 8).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ModelScopeT2VModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        StandardVAE<T>? vae = null,
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
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(videoUNet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Video U-Net and VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="videoUNet">Custom Video U-Net, or null for ModelScope defaults.</param>
    /// <param name="vae">Custom VAE, or null for standard SD VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default Video U-Net:
    /// - 4 input/output channels, 320 base channels, [1,2,4,4] multipliers
    /// - 8 attention heads, 1024-dim cross-attention
    /// - 1 temporal attention layer per block
    /// - No image conditioning support
    ///
    /// Default VAE:
    /// - Standard SD VAE with 4 latent channels
    /// - 128 base channels, 0.18215 latent scale factor
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_videoUNet), nameof(_vae))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? videoUNet,
        StandardVAE<T>? vae,
        int? seed)
    {
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM,
            numHeads: NUM_HEADS,
            numTemporalLayers: NUM_TEMPORAL_LAYERS,
            supportsImageConditioning: false);

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
        return _videoUNet.PredictNoiseWithImageCondition(
            latents, timestep, imageEmbedding, textConditioning: null);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _videoUNet.GetParameters();
        var vaeParams = _vae.GetParameters();

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
        var vaeCount = _vae.ParameterCount;

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
        return new ModelScopeT2VModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
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
            Name = "ModelScope-T2V",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "ModelScope Text-to-Video with temporal U-Net for short clip generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "temporal-unet-t2v");
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("temporal_layers", NUM_TEMPORAL_LAYERS);
        metadata.SetProperty("text_encoder", "CLIP");
        metadata.SetProperty("dataset", "WebVid-10M");
        metadata.SetProperty("scheduler", "DDPM");
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        metadata.SetProperty("open_source", true);

        return metadata;
    }

    #endregion
}
