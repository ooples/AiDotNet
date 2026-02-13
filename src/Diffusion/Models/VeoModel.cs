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
/// Veo model — Google's high-fidelity video generation with cascaded diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Veo uses cascaded diffusion with temporal super-resolution for
/// high-resolution, long-duration video generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Veo is Google's top-tier video generation model:
///
/// Key characteristics:
/// - Cascaded: base → spatial SR → temporal SR
/// - 1080p output with 60+ second duration
/// - T5-XXL + CLIP dual text encoding
/// - Flow matching training on large video corpus
///
/// Reference: Google DeepMind, "Veo: High-Fidelity Video Generation", 2024
/// </para>
/// </remarks>
public class VeoModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int HIDDEN_DIM = 2560;
    private const int NUM_LAYERS = 40;
    private const int NUM_HEADS = 20;
    private const int CONTEXT_DIM = 4096;

    private readonly DiTNoisePredictor<T> _dit;
    private readonly TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _isVeo2;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;
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
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    /// <summary>Gets whether this is a Veo 2 variant.</summary>
    public bool IsVeo2 => _isVeo2;

    public VeoModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null, bool isVeo2 = false,
        int defaultNumFrames = 150, int defaultFPS = 24)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               defaultNumFrames, defaultFPS)
    {
        _isVeo2 = isVeo2;
        _conditioner = conditioner;
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 2, contextDim: CONTEXT_DIM);
        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3, temporalKernelSize: 3,
            causalMode: true, latentScaleFactor: 0.13025);
    }

    /// <summary>Creates a Veo 2 variant.</summary>
    public static VeoModel<T> CreateVeo2(IConditioningModule<T>? conditioner = null)
        => new(isVeo2: true, conditioner: conditioner, defaultNumFrames: 200, defaultFPS: 24);

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
        => _dit.PredictNoise(latents, timestep, imageEmbedding);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters(); var vp = _temporalVAE.GetParameters();
        var c = new Vector<T>(dp.Length + vp.Length);
        for (int i = 0; i < dp.Length; i++) c[i] = dp[i];
        for (int i = 0; i < vp.Length; i++) c[dp.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int dc = _dit.ParameterCount, vc = _temporalVAE.GetParameters().Length;
        var dp = new Vector<T>(dc); var vp = new Vector<T>(vc);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[dc + i];
        _dit.SetParameters(dp); _temporalVAE.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cd = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 2, contextDim: CONTEXT_DIM);
        cd.SetParameters(_dit.GetParameters());
        return new VeoModel<T>(dit: cd,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner, isVeo2: _isVeo2,
            defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _isVeo2 ? "Veo-2" : "Veo", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = $"Google {(_isVeo2 ? "Veo 2" : "Veo")} cascaded video generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "cascaded-dit");
        m.SetProperty("is_veo2", _isVeo2);
        m.SetProperty("max_resolution", "1080p");
        return m;
    }
}
