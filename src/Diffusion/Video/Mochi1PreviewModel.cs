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
/// Mochi 1 Preview with Asymmetric Diffusion Transformer (AsymmDiT).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Mochi 1: A New SOTA in Open-Source Video Generation" (Genmo, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Mochi 1 Preview uses an Asymmetric Diffusion Transformer (AsymmDiT) that processes text and video with different-sized networks. This asymmetry efficiently handles rich text understanding for prompt adherence while maintaining high video quality.</para>
/// <para>
/// Mochi 1 Preview is a 10B parameter model using the novel Asymmetric Diffusion Transformer
/// (AsymmDiT) architecture, which uses different computational blocks for the conditioning
/// and generation pathways. This asymmetry enables efficient processing of text conditions while
/// maximizing video generation quality. Generates 480p video at 30 FPS.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: AsymmDiT (Asymmetric Diffusion Transformer)
/// - Latent channels: 12
/// - Default: 163 frames at 30 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
public class Mochi1PreviewModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 12;
    private const int CONTEXT_DIM = 4096;
    private const int DEFAULT_NUM_FRAMES = 163;
    private const int DEFAULT_FPS = 30;

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
    /// Initializes a new instance of Mochi1PreviewModel with full customization support.
    /// </summary>
    public Mochi1PreviewModel(
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
            hiddenSize: 3072,
            numLayers: 48,
            numHeads: 24,
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
            hiddenSize: 3072,
            numLayers: 48,
            numHeads: 24,
            patchSize: 2,
            contextDim: CONTEXT_DIM);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        return new Mochi1PreviewModel<T>(
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
            Name = "Mochi-1-Preview",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Mochi 1 Preview with Asymmetric Diffusion Transformer (AsymmDiT).",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "asymm-dit");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
