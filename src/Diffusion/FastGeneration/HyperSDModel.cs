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
/// Hyper-SD model for unified 1-8 step generation via trajectory-segmented distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Hyper-SD uses a novel trajectory-segmented consistency distillation that divides the
/// denoising trajectory into segments and distills each segment independently. Combined
/// with human feedback learning, it achieves state-of-the-art quality across 1-8 step
/// configurations for both SD 1.5 and SDXL architectures.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most fast-generation models are optimized for a specific step
/// count (1-step or 4-step). Hyper-SD works well across ALL step counts from 1 to 8,
/// letting you smoothly trade speed for quality. It's like having multiple specialized
/// models in one — just change the step count.
/// </para>
/// <para>
/// Reference: Ren et al., "Hyper-SD: Trajectory Segmented Consistency Model", 2024
/// </para>
/// </remarks>
public class HyperSDModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 0.0;

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
    /// Gets whether this is the SDXL variant.
    /// </summary>
    public bool IsXLVariant => _isXLVariant;

    /// <summary>
    /// Initializes a new Hyper-SD model.
    /// </summary>
    /// <param name="isXLVariant">Whether to use SDXL architecture (default: true).</param>
    public HyperSDModel(
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
        var clone = new HyperSDModel<T>(
            conditioner: _conditioner, isXLVariant: _isXLVariant, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var variant = _isXLVariant ? "SDXL" : "SD 1.5";
        var m = new ModelMetadata<T>
        {
            Name = $"Hyper-SD ({variant})", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = $"Trajectory-segmented consistency model for unified 1-8 step {variant} generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "trajectory-segmented-consistency");
        m.SetProperty("optimal_steps", 4);
        m.SetProperty("max_recommended_steps", 8);
        m.SetProperty("is_xl_variant", _isXLVariant);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
