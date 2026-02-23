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

namespace AiDotNet.Diffusion.Video.LongVideo;

/// <summary>
/// Loong autoregressive LLM-based minute-long video generator.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Loong: Generating Minute-level Long Videos with Autoregressive Language Models" (2024)</item></list></para>
/// <para><b>For Beginners:</b> Loong uses a large language model approach to generate minute-long videos by treating video frames as tokens in a sequence. This autoregressive approach naturally handles long-range temporal dependencies.</para>
/// <para>
/// Loong generates minute-long videos using an autoregressive LLM-based approach with progressive
/// short-to-long training. It addresses the loss imbalance problem for long video training through
/// a novel loss re-weighting scheme. The model treats video generation as next-token prediction
/// over visual tokens.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Autoregressive LLM + Progressive Short-to-Long Training
/// - Latent channels: 16
/// - Default: 480 frames at 8 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
public class LoongModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int CONTEXT_DIM = 4096;
    private const int DEFAULT_NUM_FRAMES = 480;
    private const int DEFAULT_FPS = 8;

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
    /// Initializes a new instance of LoongModel with full customization support.
    /// </summary>
    public LoongModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? predictor = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS,
        int? seed = null)
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
        InitializeLayers(predictor, temporalVAE, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_temporalVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? predictor,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        _predictor = predictor ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2048,
            numLayers: 24,
            numHeads: 16,
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
            hiddenSize: 2048,
            numLayers: 24,
            numHeads: 16,
            patchSize: 2,
            contextDim: CONTEXT_DIM);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        return new LoongModel<T>(
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
            Name = "Loong",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Loong autoregressive LLM-based minute-long video generator.",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "llm-autoregressive-long");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
