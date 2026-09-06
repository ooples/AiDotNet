using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Video.LongVideo;

/// <summary>
/// Snap Video scaled spatiotemporal transformer for text-to-video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis" (Snap, 2024)</item></list></para>
/// <para><b>For Beginners:</b> SnapVideo from Snap uses a scaled spatiotemporal transformer that processes space and time jointly for high-quality video generation. Its efficient architecture scales well to longer durations.</para>
/// <para>
/// Snap Video uses a scaled spatiotemporal transformer architecture for efficient text-to-video
/// synthesis. The model scales transformer attention across both spatial and temporal dimensions
/// with efficient factorization strategies. Achieves strong quality with efficient generation.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Scaled Spatiotemporal Transformer
/// - Latent channels: 16
/// - Default: 48 frames at 24 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 16, DefaultInferenceSteps = 50 };
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 16, 48, 64, 64 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new SnapVideoModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var video = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.VideoGeneration)]
[ModelTask(ModelTask.TextToVideo)]
[ModelTask(ModelTask.ImageToVideo)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis", "https://arxiv.org/abs/2402.14797", Year = 2024, Authors = "Menapace et al.")]
public partial class SnapVideoModel<T> : VideoDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_temporalVAE);
    }

    private const int LATENT_CHANNELS = 16;
    private const int CONTEXT_DIM = 4096;
    private const int DEFAULT_NUM_FRAMES = 48;
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

    /// <summary>
    /// Initializes a new instance of SnapVideoModel with full customization support.
    /// </summary>
    public SnapVideoModel(
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
            numLayers: 28,
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



    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "SnapVideo",
            Version = "1.0",
            Description = "Snap Video scaled spatiotemporal transformer for text-to-video.",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "scaled-spatiotemporal-transformer");
        metadata.SetProperty("open_source", false);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
