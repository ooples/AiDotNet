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
/// Make-A-Video model — text-to-video generation without paired text-video data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Make-A-Video leverages text-to-image models and unsupervised video learning to
/// generate videos without requiring paired text-video training data. It uses a three-stage
/// pipeline: text-to-image base generation, temporal extension for motion, and spatial plus
/// temporal super-resolution for high-quality output.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Pseudo-3D U-Net with temporal convolutions and attention</description></item>
/// <item><description>Standard VAE for spatial latent encoding (4-channel latent space)</description></item>
/// <item><description>CLIP + BPE dual text encoder for conditioning (768-dim context)</description></item>
/// <item><description>Three-stage cascade: base T2I, temporal extension, spatial/temporal SR</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Make-A-Video is Meta's video generation model that creates videos
/// from text descriptions without needing paired text-video training data.
///
/// How Make-A-Video works:
/// 1. Text prompt is encoded using CLIP and BPE into a 768-dimensional embedding
/// 2. A pseudo-3D U-Net generates an initial low-resolution video in latent space
/// 3. Temporal layers extend single images into coherent video sequences
/// 4. Spatial and temporal super-resolution stages increase quality and frame rate
///
/// Advantages:
/// - Does not require paired text-video training data
/// - Leverages existing high-quality text-to-image knowledge
/// - Supports both text-to-video and image-to-video generation
/// - Pseudo-3D convolutions are more memory-efficient than full 3D
///
/// Limitations:
/// - Lower temporal consistency than full 3D attention models
/// - Limited to shorter video clips (16 frames default)
/// - Quality depends heavily on the underlying text-to-image model
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Pseudo-3D U-Net with temporal conv and attention
/// - Latent space: 4 channels with standard VAE
/// - Text encoder: CLIP + BPE (768-dimensional context)
/// - Base resolution: 64x64 latent (256x256 pixel equivalent)
/// - Super-resolution: up to 768x768 pixels
/// - Default: 16 frames at 8 FPS
/// - Noise schedule: DDPM linear
///
/// Reference: Singer et al., "Make-A-Video: Text-to-Video Generation without Text-Video Data", ICLR 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with default settings
/// var makeAVideo = new MakeAVideoModel&lt;float&gt;();
///
/// // Generate video from text
/// var frames = makeAVideo.GenerateFromText(
///     prompt: "A teddy bear painting a portrait",
///     numFrames: 16,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
///
/// // Generate video from an image
/// var animatedFrames = makeAVideo.GenerateFromImage(
///     inputImage: referenceImage,
///     numFrames: 16);
/// </code>
/// </example>
public class MakeAVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels for the standard VAE (4).
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension from the CLIP text encoder (768).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default guidance scale for Make-A-Video (7.5).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Number of attention heads in the U-Net (8).
    /// </summary>
    private const int NUM_HEADS = 8;

    /// <summary>
    /// Base channel count for the U-Net backbone (320).
    /// </summary>
    private const int BASE_CHANNELS = 320;

    #endregion

    #region Fields

    private VideoUNetPredictor<T> _videoUNet;
    private StandardVAE<T> _vae;
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
    public override bool SupportsImageToVideo => true;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;

    /// <inheritdoc />
    public override int ParameterCount => _videoUNet.GetParameters().Length + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of MakeAVideoModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Diffusion model options. If null, uses Make-A-Video defaults.</param>
    /// <param name="scheduler">Noise scheduler. If null, uses DDPM linear scheduler.</param>
    /// <param name="videoUNet">Custom video U-Net noise predictor. If null, creates default pseudo-3D U-Net.</param>
    /// <param name="vae">Custom standard VAE. If null, creates Make-A-Video's default.</param>
    /// <param name="conditioner">Text conditioning module.</param>
    /// <param name="defaultNumFrames">Default number of frames to generate (default: 16).</param>
    /// <param name="defaultFPS">Default frames per second (default: 8).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public MakeAVideoModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 16,
        int defaultFPS = 8,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
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
    /// Initializes the video U-Net and VAE layers using custom or default configurations.
    /// </summary>
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
            numTemporalLayers: 1,
            supportsImageConditioning: true);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
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
        return _videoUNet.PredictNoiseWithImageCondition(latents, timestep, imageEmbedding, textConditioning: null);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _videoUNet.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);

        for (int i = 0; i < unetParams.Length; i++)
            combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[unetParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _videoUNet.GetParameters().Length;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[unetCount + i];

        _videoUNet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        return new MakeAVideoModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            vae: new StandardVAE<T>(
                inputChannels: 3,
                latentChannels: LATENT_CHANNELS,
                baseChannels: 128,
                channelMultipliers: [1, 2, 4, 4],
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
            Name = "Make-A-Video",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Make-A-Video text-to-video generation without paired text-video data by Meta",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "pseudo3d-unet");
        metadata.SetProperty("backbone", "Pseudo3D-UNet-320B");
        metadata.SetProperty("no_paired_data", true);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("max_resolution", "768x768");
        metadata.SetProperty("noise_schedule", "ddpm_linear");
        metadata.SetProperty("text_encoder", "CLIP+BPE");

        return metadata;
    }

    #endregion
}
