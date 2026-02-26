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
/// SD3 Flash for ultra-fast 1-4 step generation from SD3 architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SD3 Flash is the consistency-distilled variant of Stable Diffusion 3 optimized for
/// 1-4 step inference. Preserves SD3's strong text understanding and image quality
/// while dramatically reducing the number of required inference steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> SD3 Flash is to SD3 what FLUX Schnell is to FLUX — the same
/// great model but distilled to run in just 1-4 steps instead of 28. Perfect for
/// applications where speed matters more than squeezing out every bit of quality.
/// </para>
/// </remarks>
public class SD3FlashModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const double DEFAULT_GUIDANCE = 0.0;

    private MMDiTXNoisePredictor<T> _predictor;
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

    public SD3FlashModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTXNoisePredictor<T>? predictor = null,
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
    private void InitializeLayers(MMDiTXNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(
            variant: MMDiTXVariant.Medium,
            inputChannels: LATENT_CHANNELS,
            patchSize: 2,
            contextDim: 4096,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 1.5305, seed: seed);
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
        var clone = new SD3FlashModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "SD3 Flash", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Consistency-distilled SD3 for ultra-fast 1-4 step generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "distilled-mmdit-consistency");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
