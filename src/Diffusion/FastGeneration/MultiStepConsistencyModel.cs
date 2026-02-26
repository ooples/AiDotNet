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

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// Multi-Step Consistency Model that bridges single-step and multi-step generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extends consistency models to support configurable multi-step generation with improved
/// quality. While standard consistency models map noisy→clean in one step, this variant
/// chains multiple consistency steps with intermediate noise injection for better quality
/// at the cost of slightly more compute.
/// </para>
/// <para>
/// <b>For Beginners:</b> Single-step generation is fast but can lack detail. This model
/// lets you trade speed for quality: use 1 step for real-time previews, 2-4 steps for
/// high-quality results. Each additional step refines the image, similar to how an artist
/// might do a rough sketch first, then add details in subsequent passes.
/// </para>
/// <para>
/// Reference: Based on Consistency Models (Song et al., 2023) with multi-step extensions
/// </para>
/// </remarks>
public class MultiStepConsistencyModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    private const int MSCM_LATENT_CHANNELS = 4;
    private const double MSCM_DEFAULT_GUIDANCE = 1.5;

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
    public override int LatentChannels => MSCM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new Multi-Step Consistency Model.
    /// </summary>
    public MultiStepConsistencyModel(
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
        SetGuidanceScale(MSCM_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: MSCM_LATENT_CHANNELS,
            outputChannels: MSCM_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: 768,
            architecture: Architecture,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: MSCM_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Parameters

    /// <inheritdoc />
    public override Vector<T> GetParameters() => _predictor.GetParameters();

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) => _predictor.SetParameters(parameters);

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new MultiStepConsistencyModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Multi-Step Consistency Model",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Configurable multi-step consistency model bridging single-step speed and multi-step quality",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        m.SetProperty("architecture", "multi-step-consistency-unet");
        m.SetProperty("optimal_steps", 2);
        m.SetProperty("max_recommended_steps", 8);
        m.SetProperty("guidance_scale", MSCM_DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", MSCM_LATENT_CHANNELS);
        return m;
    }

    #endregion
}
