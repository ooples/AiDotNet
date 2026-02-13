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
/// ModelScope Text-to-Video model — U-Net with temporal convolutions and attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ModelScope T2V extends the text-to-image U-Net with temporal convolution and
/// temporal attention modules for generating short video clips from text.
/// </para>
/// <para>
/// <b>For Beginners:</b> ModelScope T2V generates short videos from text prompts:
///
/// Key characteristics:
/// - Extends SD U-Net with temporal blocks
/// - DDPM training on WebVid-10M dataset
/// - 256x256 → 512x512 cascaded generation
/// - CLIP text encoder for conditioning
///
/// Reference: Wang et al., "ModelScope Text-to-Video Technical Report", 2023
/// </para>
/// </remarks>
public class ModelScopeT2VModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 1024;

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
    public override bool SupportsImageToVideo => false;
    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;
    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;
    /// <inheritdoc />
    public override int ParameterCount => _videoUNet.GetParameters().Length + _vae.ParameterCount;

    public ModelScopeT2VModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null, StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 16, int defaultFPS = 8)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
               defaultNumFrames, defaultFPS)
    {
        _conditioner = conditioner;
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM, numHeads: 8,
            numTemporalLayers: 1, supportsImageConditioning: false);
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
        => new ModelScopeT2VModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            vae: new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
                baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
                numResBlocksPerLevel: 2, latentScaleFactor: 0.18215),
            conditioner: _conditioner, defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "ModelScope-T2V", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "ModelScope Text-to-Video with temporal U-Net", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "temporal-unet-t2v");
        m.SetProperty("dataset", "WebVid-10M");
        return m;
    }
}
