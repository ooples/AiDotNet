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

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// Improved Consistency Training (iCT) model for single-step image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// iCT improves upon the original Consistency Training by using a lognormal schedule for
/// the noise discretization, pseudo-Huber loss instead of L2, and an improved EMA schedule.
/// Achieves state-of-the-art FID scores among single-step generators on ImageNet 64x64.
/// </para>
/// <para>
/// <b>For Beginners:</b> Consistency models learn to map any noisy image directly to the
/// clean image in a single step. Unlike diffusion models that remove noise gradually over
/// 20-50 steps, iCT does it in one shot. The "improved" version trains better by using
/// smarter loss functions and noise schedules, producing higher quality single-step images.
/// </para>
/// <para>
/// Reference: Song and Dhariwal, "Improved Techniques for Training Consistency Models", ICLR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 1 };
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 8, 8 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new ImprovedConsistencyModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var generated = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Improved Techniques for Training Consistency Models", "https://arxiv.org/abs/2310.14189", Year = 2024, Authors = "Song and Dhariwal")]
public partial class ImprovedConsistencyModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    #region Constants

    private const int ICT_LATENT_CHANNELS = 4;
    private const double ICT_DEFAULT_GUIDANCE = 0.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => ICT_LATENT_CHANNELS;

    /// <inheritdoc />

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new Improved Consistency Training model.
    /// </summary>
    public ImprovedConsistencyModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(ICT_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: ICT_LATENT_CHANNELS,
            outputChannels: ICT_LATENT_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: 768,
            architecture: Architecture,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: ICT_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Parameters



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Improved Consistency Training (iCT)",
            Version = "1.0",
            Description = "Single-step image generation via improved consistency training with lognormal schedule and pseudo-Huber loss",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        m.SetProperty("architecture", "consistency-model-unet");
        m.SetProperty("training_method", "improved-consistency-training");
        m.SetProperty("loss_function", "pseudo-Huber");
        m.SetProperty("noise_schedule", "lognormal");
        m.SetProperty("optimal_steps", 1);
        m.SetProperty("max_recommended_steps", 2);
        m.SetProperty("guidance_scale", ICT_DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", ICT_LATENT_CHANNELS);
        return m;
    }

    #endregion
}
