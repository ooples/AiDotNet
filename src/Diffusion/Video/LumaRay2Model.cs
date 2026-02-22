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
/// Luma Ray 2 video model with fast natural motion and better physics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Reference: Luma AI Ray 2 (2025)</item></list></para>
/// <para><b>For Beginners:</b> Luma Ray 2 produces videos with fast, natural motion using 10x more compute than its predecessor. It excels at realistic movement patterns and physical interactions between objects.</para>
/// <para>
/// Ray 2 was trained with 10x more compute than Ray1, offering fast, natural coherent motion and
/// physics understanding. The model generates videos with natural motion dynamics and good temporal
/// consistency. Features improved physics simulation for realistic object interactions.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT + Enhanced Physics + 10x Compute
/// - Latent channels: 16
/// - Default: 120 frames at 24 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
public class LumaRay2Model<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int CONTEXT_DIM = 4096;
    private const int DEFAULT_NUM_FRAMES = 120;
    private const int DEFAULT_FPS = 24;

    private DiTNoisePredictor<T> _predictor;
    private TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _temporalVAE;
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override bool SupportsImageToVideo => true;
    public override bool SupportsTextToVideo => true;
    public override bool SupportsVideoToVideo => false;
    public override int ParameterCount => _predictor.ParameterCount + _temporalVAE.GetParameters().Length;

    /// <summary>
    /// Initializes a new instance of LumaRay2Model with full customization support.
    /// </summary>
    public LumaRay2Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? predictor = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, temporalVAE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_temporalVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? predictor,
        TemporalVAE<T>? temporalVAE)
    {
        _predictor = predictor ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2560,
            numLayers: 32,
            numHeads: 20,
            patchSize: 2,
            contextDim: CONTEXT_DIM);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3,
            temporalKernelSize: 3,
            causalMode: true,
            latentScaleFactor: 0.13025);
    }

    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        return _predictor.PredictNoise(latents, timestep, imageEmbedding);
    }

    public override Vector<T> GetParameters()
    {
        var predParams = _predictor.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();
        var combined = new Vector<T>(predParams.Length + vaeParams.Length);
        for (int i = 0; i < predParams.Length; i++) combined[i] = predParams[i];
        for (int i = 0; i < vaeParams.Length; i++) combined[predParams.Length + i] = vaeParams[i];
        return combined;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        var predCount = _predictor.ParameterCount;
        var vaeCount = _temporalVAE.GetParameters().Length;
        if (parameters.Length != predCount + vaeCount)
            throw new ArgumentException($"Expected {predCount + vaeCount} parameters, got {parameters.Length}.", nameof(parameters));
        var predParams = new Vector<T>(predCount);
        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < predCount; i++) predParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++) vaeParams[i] = parameters[predCount + i];
        _predictor.SetParameters(predParams);
        _temporalVAE.SetParameters(vaeParams);
    }

    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    public override IDiffusionModel<T> Clone()
    {
        var clonedPredictor = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2560,
            numLayers: 32,
            numHeads: 20,
            patchSize: 2,
            contextDim: CONTEXT_DIM);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        return new LumaRay2Model<T>(
            architecture: Architecture,
            options: Options as DiffusionModelOptions<T>,
            scheduler: Scheduler,
            predictor: clonedPredictor,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Luma-Ray-2",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Luma Ray 2 video model with fast natural motion and better physics.",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "dit-physics-enhanced");
        metadata.SetProperty("open_source", false);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
