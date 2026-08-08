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

namespace AiDotNet.Diffusion.Video;

/// <summary>
/// Open-Sora 2.0 commercial-level video generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k" (HPC-AI Tech, 2025)</item></list></para>
/// <para><b>For Beginners:</b> Open-Sora 2.0 is a commercial-grade open-source video generator trained for under $200K. It produces high-quality videos competitive with proprietary models, using an improved architecture with better temporal modeling and data curation.</para>
/// <para>
/// Open-Sora 2.0 achieves commercial-level video quality comparable to HunyuanVideo and Runway Gen-3
/// alpha while being trained for only $200k. Uses an improved STDiT backbone with enhanced temporal
/// attention, better VAE, and data curation pipeline. Human evaluation shows it matches or exceeds
/// leading commercial models.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Enhanced STDiT + Improved 3D-VAE + Rectified Flow
/// - Latent channels: 16
/// - Default: 93 frames at 24 FPS
/// - Supports I2V: Yes | T2V: Yes | V2V: No
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 16, Height = 720, Width = 1280, NumInferenceSteps = 30 };
/// var model = new OpenSora2Model&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 16, 93, 90, 160 });
/// var video = model.Predict(noise);
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
[ResearchPaper("Open-Sora: Democratizing Efficient Video Production for All", "https://arxiv.org/abs/2412.20404", Year = 2024, Authors = "Zheng et al.")]
public class OpenSora2Model<T> : VideoDiffusionModelBase<T>
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
    private const int DEFAULT_NUM_FRAMES = 93;
    private const int DEFAULT_FPS = 24;

    private DiTNoisePredictor<T>? _predictor;
    private TemporalVAE<T>? _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;
    // Seed for the deferred (lazy) init path: the constructor only eager-inits when an explicit
    // predictor/VAE is passed, so without capturing the seed the lazy EnsureInitialized() built the
    // sub-models with a null seed — dropping the requested seed and making construction non-reproducible.
    private readonly int? _seed;

    public override INoisePredictor<T> NoisePredictor { get { EnsureInitialized(); return _predictor; } }
    public override IVAEModel<T> VAE { get { EnsureInitialized(); return _temporalVAE; } }
    public override IVAEModel<T>? TemporalVAE { get { EnsureInitialized(); return _temporalVAE; } }
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override bool SupportsImageToVideo => true;
    public override bool SupportsTextToVideo => true;
    public override bool SupportsVideoToVideo => false;


    /// <summary>
    /// Initializes a new instance of OpenSora2Model with full customization support.
    /// </summary>
    public OpenSora2Model(
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
        _seed = seed;
        if (predictor is not null || temporalVAE is not null)
            InitializeLayers(predictor, temporalVAE, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_temporalVAE))]
    private void EnsureInitialized()
    {
        if (_predictor is null || _temporalVAE is null)
            InitializeLayers(null, null, _seed);
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
            numLayers: 32,
            numHeads: 16,
            patchSize: 2,
            contextDim: CONTEXT_DIM,
            seed: seed);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 4,
            temporalKernelSize: 3,
            causalMode: true,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        EnsureInitialized();
        return _predictor.PredictNoise(latents, timestep, imageEmbedding);
    }



    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    public override IDiffusionModel<T> Clone()
    {
        EnsureInitialized();
                return new OpenSora2Model<T>(
            predictor: (DiTNoisePredictor<T>)_predictor.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "OpenSora2",
            Version = "2.0",
            Description = "Open-Sora 2.0 commercial-level video generation model.",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "stdit-v2-commercial");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);
        return metadata;
    }
}
