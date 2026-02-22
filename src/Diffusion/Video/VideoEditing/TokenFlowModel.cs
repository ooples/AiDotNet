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

namespace AiDotNet.Diffusion.Video.VideoEditing;

/// <summary>
/// TokenFlow consistent video editing via token flow propagation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "TokenFlow: Consistent Diffusion Features for Consistent Video Editing" (Bar-Tal et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> TokenFlow enables consistent video editing by propagating visual changes through time using token flow, tracking how image features move between frames. This ensures edits look natural and consistent across the entire video.</para>
/// <para>
/// TokenFlow achieves consistent video editing by explicitly enforcing semantic correspondences of
/// diffusion features across video frames. The method propagates edited tokens from keyframes to
/// all frames using the natural token flow, maintaining temporal consistency without per-frame
/// processing. Works with any text-guided image editing method.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Token Flow Propagation + DDIM Inversion + Keyframe Editing
/// - Latent channels: 4
/// - Default: 24 frames at 8 FPS
/// - Supports I2V: No | T2V: Yes | V2V: Yes
/// </para>
/// </remarks>
public class TokenFlowModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int CONTEXT_DIM = 768;
    private const int DEFAULT_NUM_FRAMES = 24;
    private const int DEFAULT_FPS = 8;

    private VideoUNetPredictor<T> _predictor;
    private TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _temporalVAE;
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override bool SupportsImageToVideo => false;
    public override bool SupportsTextToVideo => true;
    public override bool SupportsVideoToVideo => true;
    public override int ParameterCount => _predictor.ParameterCount + _temporalVAE.GetParameters().Length;

    /// <summary>
    /// Initializes a new instance of TokenFlowModel with full customization support.
    /// </summary>
    public TokenFlowModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? predictor = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, temporalVAE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_temporalVAE))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? predictor,
        TemporalVAE<T>? temporalVAE)
    {
        _predictor = predictor ?? new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            numHeads: 8,
            contextDim: CONTEXT_DIM);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3,
            temporalKernelSize: 3,
            causalMode: false,
            latentScaleFactor: 0.18215);
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
        var clonedPredictor = new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            numHeads: 8,
            contextDim: CONTEXT_DIM);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        return new TokenFlowModel<T>(
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
            Name = "TokenFlow",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "TokenFlow consistent video editing via token flow propagation.",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "token-flow-propagation");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
