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
/// Runway Gen model — latent diffusion model for high-fidelity video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Runway Gen (Gen-1/Gen-2/Gen-3) uses temporal diffusion with multi-modal conditioning
/// for photorealistic video generation and editing.
/// </para>
/// <para>
/// <b>For Beginners:</b> Runway Gen creates professional-quality videos:
///
/// Key characteristics:
/// - Multi-modal: text, image, video, and motion conditioning
/// - Temporal U-Net with cross-frame attention
/// - Structure and style disentanglement
/// - Cascaded generation for high resolution
///
/// Reference: Esser et al., "Structure and Content-Guided Video Synthesis with Diffusion Models", ICCV 2023
/// </para>
/// </remarks>
public class RunwayGenModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 1024;

    private readonly VideoUNetPredictor<T> _videoUNet;
    private readonly TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _isGen3;

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
    public override int ParameterCount => _videoUNet.GetParameters().Length + _temporalVAE.GetParameters().Length;

    /// <summary>Gets whether this is a Gen-3 variant.</summary>
    public bool IsGen3 => _isGen3;

    public RunwayGenModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null, TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null, bool isGen3 = false,
        int defaultNumFrames = 25, int defaultFPS = 24)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
               defaultNumFrames, defaultFPS)
    {
        _isGen3 = isGen3;
        _conditioner = conditioner;
        _videoUNet = videoUNet ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: isGen3 ? 384 : 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: isGen3 ? 2048 : CROSS_ATTENTION_DIM, numHeads: isGen3 ? 16 : 8,
            numTemporalLayers: isGen3 ? 3 : 1, supportsImageConditioning: true);
        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 1, temporalKernelSize: 3,
            causalMode: isGen3, latentScaleFactor: 0.18215);
    }

    /// <summary>Creates a Gen-3 Alpha variant.</summary>
    public static RunwayGenModel<T> CreateGen3Alpha(IConditioningModule<T>? conditioner = null)
        => new(isGen3: true, conditioner: conditioner, defaultNumFrames: 150, defaultFPS: 24);

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
        => _videoUNet.PredictNoiseWithImageCondition(latents, timestep, imageEmbedding, textConditioning: null);

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
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _videoUNet.SetParameters(up); _temporalVAE.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
        => new RunwayGenModel<T>(
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner, isGen3: _isGen3,
            defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _isGen3 ? "Runway-Gen-3" : "Runway-Gen-2", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = $"Runway {(_isGen3 ? "Gen-3" : "Gen-2")} video generation model", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "temporal-unet-multimodal");
        m.SetProperty("is_gen3", _isGen3);
        return m;
    }
}
