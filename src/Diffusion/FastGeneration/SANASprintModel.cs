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
/// SANA Sprint model for ultra-fast 1-step generation from the SANA architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SANA Sprint is the distilled version of SANA that generates 1024x1024 images in a
/// single step. Uses the efficient linear attention DiT backbone from SANA combined
/// with hybrid distillation (consistency + adversarial) for real-time generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> SANA is already fast due to its efficient architecture.
/// SANA Sprint makes it even faster — single-step generation at 1024x1024 resolution.
/// This makes it one of the fastest high-resolution generators available, suitable for
/// real-time applications like interactive image editing.
/// </para>
/// <para>
/// Reference: NVIDIA, "SANA Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation", 2025
/// </para>
/// </remarks>
public class SANASprintModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 32;
    private const double DEFAULT_GUIDANCE = 0.0;

    private SiTPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Initializes a new SANA Sprint model.
    /// </summary>
    public SANASprintModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        SiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(SiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2048,
            numLayers: 20,
            numHeads: 32,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.41407, seed: seed);
    }

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
        var clone = new SANASprintModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "SANA Sprint", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Ultra-fast single-step 1024x1024 generation via SANA architecture with hybrid distillation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "sana-distilled-linear-dit");
        m.SetProperty("distillation_method", "hybrid-consistency-adversarial");
        m.SetProperty("optimal_steps", 1);
        m.SetProperty("max_recommended_steps", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
