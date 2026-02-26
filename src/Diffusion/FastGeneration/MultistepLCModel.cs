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
/// Multistep Latent Consistency Model (MLCM) for high-quality few-step generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MLCM extends Latent Consistency Models to support multiple distillation targets
/// per training step, improving generation quality at 2-8 steps. Uses a multistep
/// consistency loss that enforces the model to be self-consistent across multiple
/// denoising trajectories simultaneously.
/// </para>
/// <para>
/// <b>For Beginners:</b> LCM generates images in 2-4 steps. MLCM improves on this by
/// training the model to be consistent across different numbers of steps — meaning it
/// produces good results whether you use 2, 4, or 8 steps. This flexibility lets you
/// choose the best speed/quality tradeoff for your use case.
/// </para>
/// <para>
/// Reference: Based on Latent Consistency Models (Luo et al., 2023) with multistep extensions
/// </para>
/// </remarks>
public class MultistepLCModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    private const int MLCM_LATENT_CHANNELS = 4;
    private const double MLCM_DEFAULT_GUIDANCE = 1.0;

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
    public override int LatentChannels => MLCM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new Multistep Latent Consistency Model.
    /// </summary>
    public MultistepLCModel(
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
            scheduler ?? new LCMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(MLCM_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: MLCM_LATENT_CHANNELS,
            outputChannels: MLCM_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: 768,
            architecture: Architecture,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: MLCM_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Parameters

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var pp = _predictor.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(pp.Length + vp.Length);
        for (int i = 0; i < pp.Length; i++) combined[i] = pp[i];
        for (int i = 0; i < vp.Length; i++) combined[pp.Length + i] = vp[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var pc = _predictor.ParameterCount;
        var vc = _vae.ParameterCount;
        if (parameters.Length != pc + vc)
            throw new ArgumentException($"Expected {pc + vc} parameters, got {parameters.Length}.", nameof(parameters));
        var pp = new Vector<T>(pc);
        var vp = new Vector<T>(vc);
        for (int i = 0; i < pc; i++) pp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[pc + i];
        _predictor.SetParameters(pp);
        _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new MultistepLCModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Multistep Latent Consistency Model (MLCM)",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Multistep consistency distillation for flexible few-step generation with quality scaling",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };
        m.SetProperty("architecture", "latent-consistency-unet");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("distillation_method", "multistep-consistency");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("max_recommended_steps", 8);
        m.SetProperty("guidance_scale", MLCM_DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", MLCM_LATENT_CHANNELS);
        return m;
    }

    #endregion
}
