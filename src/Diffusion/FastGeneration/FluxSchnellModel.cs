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
/// FLUX.1 Schnell for ultra-fast 1-4 step generation from the FLUX architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLUX.1 Schnell is the distilled variant of FLUX.1 optimized for 1-4 step generation.
/// Uses the same double-stream transformer architecture as FLUX.1 Dev but distilled for
/// speed. Requires no classifier-free guidance, producing high-quality images instantly.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLUX.1 is one of the best open-source image generators.
/// FLUX.1 Schnell (German for "fast") is its speed-optimized version that generates
/// images in just 1-4 steps. It's free for commercial use and produces remarkably good
/// images for a distilled model.
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.1 Schnell", 2024
/// </para>
/// </remarks>
public class FluxSchnellModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const double DEFAULT_GUIDANCE = 0.0;

    private FluxDoubleStreamPredictor<T> _predictor;
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

    public FluxSchnellModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FluxDoubleStreamPredictor<T>? predictor = null,
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
    private void InitializeLayers(FluxDoubleStreamPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(
            variant: FluxPredictorVariant.Schnell,
            inputChannels: LATENT_CHANNELS,
            contextDim: 4096,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.3611, seed: seed);
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
        var clone = new FluxSchnellModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "FLUX.1 Schnell", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Ultra-fast 1-4 step FLUX generation, Apache 2.0 licensed",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flux-double-stream-distilled");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
