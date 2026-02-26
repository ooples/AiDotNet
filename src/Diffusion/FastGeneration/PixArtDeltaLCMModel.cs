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
/// PixArt-Delta LCM for few-step generation from the efficient PixArt-Delta DiT architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Applies Latent Consistency Model distillation to PixArt-Delta, a training-efficient
/// DiT model. PixArt-Delta uses a decomposed cross-attention mechanism that significantly
/// reduces training cost. The LCM variant enables 2-8 step generation while maintaining
/// PixArt-Delta's strong text-image alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> PixArt-Delta is an efficient alternative to SD3 — it produces
/// great images with much less training data and compute. This LCM version makes it
/// even faster at inference time, generating in just 2-8 steps. Good for when you want
/// high quality but don't need FLUX/SD3-level capability.
/// </para>
/// <para>
/// Reference: Chen et al., "PixArt-delta: Fast and Controllable Image Generation with Latent Consistency Models", 2024
/// </para>
/// </remarks>
public class PixArtDeltaLCMModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
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
    public override int ParameterCount => _predictor.ParameterCount;

    public PixArtDeltaLCMModel(
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
            scheduler ?? new LCMScheduler<T>(SchedulerConfig<T>.CreateLCM()),
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
            hiddenSize: 1152,
            numLayers: 28,
            numHeads: 16,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters() => _predictor.GetParameters();
    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) => _predictor.SetParameters(parameters);
    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new PixArtDeltaLCMModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "PixArt-Delta LCM", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Latent Consistency Model distillation of PixArt-Delta for 2-8 step generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "pixart-delta-lcm-dit");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("max_recommended_steps", 8);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
