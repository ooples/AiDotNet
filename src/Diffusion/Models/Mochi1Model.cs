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
/// Mochi 1 model — asymmetric DiT with joint attention for video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mochi 1 uses an Asymmetric DiT (AsymmDiT) architecture with joint text-video
/// attention and an asymmetric encoder-decoder VAE.
/// </para>
/// <para>
/// <b>For Beginners:</b> Mochi 1 is a state-of-the-art open video model:
///
/// Key characteristics:
/// - AsymmDiT: Asymmetric Diffusion Transformer
/// - Joint text-video attention (not cross-attention)
/// - Asymmetric VAE (lightweight encoder, heavy decoder)
/// - 480p at 30fps, open-source weights
///
/// Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024
/// </para>
/// </remarks>
public class Mochi1Model<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 12;
    private const int HIDDEN_DIM = 3072;
    private const int NUM_LAYERS = 48;
    private const int NUM_HEADS = 24;
    private const int CONTEXT_DIM = 4096;

    private readonly DiTNoisePredictor<T> _dit;
    private readonly TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;

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
    public override bool SupportsImageToVideo => false;
    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;
    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;
    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    public Mochi1Model(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 84, int defaultFPS = 30)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               defaultNumFrames, defaultFPS)
    {
        _conditioner = conditioner;
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 2, contextDim: CONTEXT_DIM);
        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 2, temporalKernelSize: 3,
            causalMode: true, latentScaleFactor: 0.13025);
    }

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
        return new Mochi1Model<T>(dit: cd,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner, defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Mochi-1", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Mochi 1 asymmetric DiT video generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "asymm-dit-joint-attention");
        m.SetProperty("open_source", true);
        m.SetProperty("parameters_billions", 10);
        return m;
    }
}
