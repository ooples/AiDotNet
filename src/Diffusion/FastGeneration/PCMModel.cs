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
/// Phased Consistency Model (PCM) for flexible-step generation with phase-based training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PCM divides the diffusion trajectory into phases and enforces consistency within each
/// phase rather than across the entire trajectory. This phased approach allows the model
/// to maintain high quality across different step configurations (1, 2, 4, 8, 16 steps)
/// without the quality degradation seen in standard consistency models at higher steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard consistency models work great at 1-2 steps but lose
/// quality at higher steps. PCM fixes this by dividing the generation process into phases
/// and handling each phase separately. This means you get good results from 1 to 16 steps,
/// giving you maximum flexibility in the speed/quality tradeoff.
/// </para>
/// <para>
/// Reference: Wang et al., "Phased Consistency Model", NeurIPS 2024
/// </para>
/// </remarks>
public class PCMModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 5.0;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _isXLVariant;

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
    /// Initializes a new Phased Consistency Model.
    /// </summary>
    /// <param name="isXLVariant">Whether to use SDXL architecture (default: true).</param>
    public PCMModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        bool isXLVariant = true,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _isXLVariant = isXLVariant;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        int contextDim = _isXLVariant ? 2048 : 768;
        int[] channelMults = _isXLVariant ? [1, 2, 4] : [1, 2, 4, 4];
        int[] attnRes = _isXLVariant ? [4, 2] : [4, 2, 1];

        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: channelMults,
            numResBlocks: 2, attentionResolutions: attnRes,
            contextDim: contextDim, architecture: Architecture, seed: seed);

        double latentScale = _isXLVariant ? 0.13025 : 0.18215;
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: latentScale, seed: seed);
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
        var clone = new PCMModel<T>(
            conditioner: _conditioner, isXLVariant: _isXLVariant, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Phased Consistency Model (PCM)", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Phase-based consistency training for flexible 1-16 step generation without quality degradation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "phased-consistency-unet");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("max_recommended_steps", 16);
        m.SetProperty("is_xl_variant", _isXLVariant);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
