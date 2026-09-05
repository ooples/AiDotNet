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
/// Show-1 marrying pixel and latent diffusion for text-to-video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation" (2023)</item></list></para>
/// <para><b>For Beginners:</b> Show-1 combines pixel-space and latent-space diffusion in a two-stage pipeline. The first stage generates coarse video in pixel space, then the second refines it in latent space for high-quality output.</para>
/// <para>
/// Show-1 combines pixel-space and latent-space diffusion models in a cascaded pipeline. The pixel
/// diffusion model handles low-resolution generation for accurate motion, while the latent diffusion
/// model performs super-resolution for visual quality. This marriage leverages the strengths of both
/// approaches.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Pixel Diffusion + Latent Super-Resolution Cascade
/// - Latent channels: 4
/// - Default: 29 frames at 8 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 50 };
/// var model = new Show1Model&lt;float&gt;(options: options);
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 29, 32, 32 });
/// var video = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.VideoGeneration)]
[ModelTask(ModelTask.TextToVideo)]
[ModelTask(ModelTask.ImageToVideo)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation", "https://arxiv.org/abs/2309.15818", Year = 2023, Authors = "Zhang et al.")]
public partial class Show1Model<T> : VideoDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_temporalVAE);
    }

    private const int LATENT_CHANNELS = 4;
    private const int CONTEXT_DIM = 768;
    private const int DEFAULT_NUM_FRAMES = 29;
    private const int DEFAULT_FPS = 8;

    private VideoUNetPredictor<T> _predictor;
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
    /// Initializes a new instance of Show1Model with full customization support.
    /// </summary>
    public Show1Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? predictor = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS,
        int? seed = null)
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
        InitializeLayers(predictor, temporalVAE, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_temporalVAE))]
    private void InitializeLayers(
        VideoUNetPredictor<T>? predictor,
        TemporalVAE<T>? temporalVAE,
        int? seed)
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



    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Show1",
            Version = "1.0",
            Description = "Show-1 marrying pixel and latent diffusion for text-to-video.",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "pixel-latent-cascade");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
