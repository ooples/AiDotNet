using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// CogVideo/CogVideoX model for text-to-video and image-to-video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CogVideoX is a large-scale text-to-video generation model developed by Zhipu AI / THUDM.
/// It uses a 3D causal VAE and a transformer-based architecture for generating coherent video.
/// </para>
/// <para>
/// <b>For Beginners:</b> CogVideoX generates videos from text descriptions:
///
/// How CogVideoX works:
/// 1. Text is encoded by a T5 text encoder
/// 2. A 3D causal VAE encodes/decodes video in compressed latent space
/// 3. A diffusion transformer (DiT) denoises the video latents
/// 4. The 3D VAE decodes latents back to video frames
///
/// Key characteristics:
/// - 3D causal VAE for temporal compression (4x temporal, 8x spatial)
/// - Expert transformer blocks with adaptive layer norm
/// - T5 text encoder for text understanding
/// - CogVideoX-2B: 2B parameters, 480p output
/// - CogVideoX-5B: 5B parameters, 720p output
/// - 49 frames at 8 FPS (~6 seconds)
///
/// Advantages:
/// - Open-source with permissive license
/// - Strong temporal coherence
/// - Good prompt adherence
/// - Efficient 3D VAE compression
///
/// Limitations:
/// - Generation is slow (~minutes per video)
/// - Requires significant VRAM
/// - Limited to short clips
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: 3D Causal VAE + Diffusion Transformer
/// - CogVideoX-2B: 2B parameters, 480×720
/// - CogVideoX-5B: 5B parameters, 480×720
/// - 3D Causal VAE: 4x temporal + 8x spatial compression, 16 latent channels
/// - Text encoder: T5-XXL (4096-dim)
/// - Frames: 49 at 8 FPS
/// - Noise schedule: Scaled linear, 1000 training timesteps
///
/// Reference: Hong et al., "CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers", ICLR 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var cogvideo = new CogVideoModel&lt;float&gt;();
/// var video = cogvideo.GenerateFromText(
///     prompt: "A cat playing piano",
///     numFrames: 49,
///     fps: 8,
///     numInferenceSteps: 50,
///     guidanceScale: 6.0,
///     seed: 42);
/// </code>
/// </example>
public class CogVideoModel<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 720;
    public const int DefaultHeight = 480;

    private const int COG_LATENT_CHANNELS = 16;
    private const int COG_CROSS_ATTENTION_DIM = 4096;
    private const double COG_DEFAULT_GUIDANCE_SCALE = 6.0;

    #endregion

    #region Fields

    private VideoUNetPredictor<T> _videoUnet;
    private TemporalVAE<T> _temporalVae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly string _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _videoUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _temporalVae;

    /// <inheritdoc />
    public override IVAEModel<T>? TemporalVAE => _temporalVae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => COG_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsImageToVideo => true;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => _conditioner != null;

    /// <inheritdoc />
    public override int ParameterCount => _videoUnet.ParameterCount + _temporalVae.ParameterCount;

    /// <summary>
    /// Gets the model variant ("2B" or "5B").
    /// </summary>
    public string Variant => _variant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of CogVideoModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses CogVideoX defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="videoUnet">Custom video U-Net/transformer predictor.</param>
    /// <param name="temporalVae">Custom 3D causal VAE.</param>
    /// <param name="conditioner">Text encoder conditioning module (typically T5-XXL).</param>
    /// <param name="variant">Model variant: "2B" or "5B" (default: "5B").</param>
    /// <param name="defaultNumFrames">Default number of frames (default: 49).</param>
    /// <param name="defaultFPS">Default FPS (default: 8).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public CogVideoModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUnet = null,
        TemporalVAE<T>? temporalVae = null,
        IConditioningModule<T>? conditioner = null,
        string variant = "5B",
        int defaultNumFrames = 49,
        int defaultFPS = 8,
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
        _variant = variant;

        InitializeLayers(videoUnet, temporalVae, seed);

        SetGuidanceScale(COG_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_videoUnet), nameof(_temporalVae))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? videoUnet,
        TemporalVAE<T>? temporalVae,
        int? seed)
    {
        var baseChannels = _variant == "5B" ? 384 : 320;

        _videoUnet = videoUnet ?? new VideoUNetPredictor<T>(
            inputChannels: COG_LATENT_CHANNELS,
            baseChannels: baseChannels,
            contextDim: COG_CROSS_ATTENTION_DIM,
            numTemporalLayers: 2,
            seed: seed);

        _temporalVae = temporalVae ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: COG_LATENT_CHANNELS,
            baseChannels: 128,
            numTemporalLayers: 2,
            causalMode: true,
            seed: seed);
    }

    #endregion

    #region Video Noise Prediction

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        return _videoUnet.PredictNoise(latents, timestep, imageEmbedding);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _videoUnet.GetParameters();
        var vaeParams = _temporalVae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
            combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[unetParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _videoUnet.ParameterCount;
        var vaeCount = _temporalVae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[unetCount + i];

        _videoUnet.SetParameters(unetParams);
        _temporalVae.SetParameters(vaeParams);
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
        var baseChannels = _variant == "5B" ? 384 : 320;

        var clonedUnet = new VideoUNetPredictor<T>(
            inputChannels: COG_LATENT_CHANNELS,
            baseChannels: baseChannels,
            contextDim: COG_CROSS_ATTENTION_DIM,
            numTemporalLayers: 2);
        clonedUnet.SetParameters(_videoUnet.GetParameters());

        var clonedVae = new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: COG_LATENT_CHANNELS,
            baseChannels: 128,
            numTemporalLayers: 2,
            causalMode: true);
        clonedVae.SetParameters(_temporalVae.GetParameters());

        return new CogVideoModel<T>(
            videoUnet: clonedUnet,
            temporalVae: clonedVae,
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
        var metadata = new ModelMetadata<T>
        {
            Name = $"CogVideoX-{_variant}",
            Version = _variant,
            ModelType = ModelType.NeuralNetwork,
            Description = $"CogVideoX-{_variant} text-to-video model with 3D causal VAE and diffusion transformer",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "3d-causal-vae-dit");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("cross_attention_dim", COG_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", COG_LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DefaultNumFrames);
        metadata.SetProperty("default_fps", DefaultFPS);
        metadata.SetProperty("resolution", $"{DefaultHeight}x{DefaultWidth}");

        return metadata;
    }

    #endregion
}
