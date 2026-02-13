using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Upscale-A-Video model — temporally consistent video super-resolution with diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Upscale-A-Video extends image super-resolution to video with temporal consistency,
/// using temporal layers and flow-guided propagation for flicker-free upscaling.
/// </para>
/// <para>
/// <b>For Beginners:</b> Upscale-A-Video increases video resolution without flickering:
///
/// Key characteristics:
/// - 4x video upscaling with temporal consistency
/// - Temporal attention layers prevent inter-frame flickering
/// - Flow-guided feature propagation for coherent motion
/// - Built on SD architecture with temporal extensions
///
/// Reference: Zhou et al., "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution", CVPR 2024
/// </para>
/// </remarks>
public class UpscaleAVideoModel<T> : VideoDiffusionModelBase<T>
{
    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 576;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 1024;

    private readonly VideoUNetPredictor<T> _videoUNet;
    private readonly TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;

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
    public override int ParameterCount => _videoUNet.GetParameters().Length + _temporalVAE.GetParameters().Length;

    public UpscaleAVideoModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null, TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null, int defaultNumFrames = 16, int defaultFPS = 24)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
               defaultNumFrames, defaultFPS)
    {
        _conditioner = conditioner;
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS * 2, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM, numHeads: 8,
            numTemporalLayers: 2, supportsImageConditioning: true);
        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 1, temporalKernelSize: 3,
            causalMode: false, latentScaleFactor: 0.18215);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
    {
        return _videoUNet.PredictNoiseWithImageCondition(
            latents, timestep, imageEmbedding, textConditioning: null);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _videoUNet.GetParameters(); var vp = _temporalVAE.GetParameters();
        var c = new Vector<T>(up.Length + vp.Length);
        for (int i = 0; i < up.Length; i++) c[i] = up[i];
        for (int i = 0; i < vp.Length; i++) c[up.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _videoUNet.GetParameters().Length, vc = _temporalVAE.GetParameters().Length;
        if (parameters.Length != uc + vc) throw new ArgumentException($"Expected {uc + vc}, got {parameters.Length}.", nameof(parameters));
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _videoUNet.SetParameters(up); _temporalVAE.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        return new UpscaleAVideoModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Upscale-A-Video", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Upscale-A-Video temporally consistent video super-resolution", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "temporal-sr-diffusion");
        m.SetProperty("upscale_factor", 4);
        m.SetProperty("temporal_consistency", true);
        return m;
    }
}
