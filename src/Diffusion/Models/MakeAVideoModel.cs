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
/// Make-A-Video model — text-to-video generation without paired text-video data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Make-A-Video leverages text-to-image models and unsupervised video learning to
/// generate videos without requiring paired text-video training data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Make-A-Video generates video from text using image knowledge:
///
/// Key characteristics:
/// - Three-stage: T2I base, temporal extension, spatial/temporal SR
/// - No paired text-video data required for training
/// - Pseudo-3D convolutions and attention
/// - CLIP + BPE text encoding
///
/// Reference: Singer et al., "Make-A-Video: Text-to-Video Generation without Text-Video Data", ICLR 2023
/// </para>
/// </remarks>
public class MakeAVideoModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;

    private readonly VideoUNetPredictor<T> _videoUNet;
    private readonly StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

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

    public MakeAVideoModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null, StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 16, int defaultFPS = 8)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               defaultNumFrames, defaultFPS)
    {
        _conditioner = conditioner;
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM, numHeads: 8,
            numTemporalLayers: 1, supportsImageConditioning: true);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
        => _videoUNet.PredictNoiseWithImageCondition(latents, timestep, imageEmbedding, textConditioning: null);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _videoUNet.GetParameters(); var vp = _vae.GetParameters();
        var c = new Vector<T>(up.Length + vp.Length);
        for (int i = 0; i < up.Length; i++) c[i] = up[i];
        for (int i = 0; i < vp.Length; i++) c[up.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _videoUNet.GetParameters().Length, vc = _vae.ParameterCount;
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _videoUNet.SetParameters(up); _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
        => new MakeAVideoModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            vae: new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
                baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
                numResBlocksPerLevel: 2, latentScaleFactor: 0.18215),
            conditioner: _conditioner, defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Make-A-Video", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Make-A-Video text-to-video without paired data", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "pseudo3d-unet");
        m.SetProperty("no_paired_data", true);
        return m;
    }
}
