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

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// PixArt-Delta model — LCM-distilled PixArt for fast 2-8 step generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PixArt-Delta applies Latent Consistency Model (LCM) distillation to PixArt-Alpha,
/// enabling high-quality image generation in just 2-8 denoising steps while preserving
/// the efficient DiT architecture.
/// </para>
/// <para>
/// <b>For Beginners:</b> PixArt-Delta is PixArt-Alpha made faster:
///
/// Key characteristics:
/// - Same DiT architecture as PixArt-Alpha
/// - LCM distillation: trained to produce good results in 2-8 steps
/// - No classifier-free guidance needed (saves 50% compute)
/// - Maintains T5-XXL text encoder for good prompt understanding
///
/// How it compares:
/// - PixArt-Alpha: ~20 steps, needs CFG → slower
/// - PixArt-Delta: ~4 steps, no CFG needed → 10x faster
/// - SD Turbo: 1-4 steps but lower quality
/// - PixArt-Delta: 2-8 steps with higher quality
///
/// Use PixArt-Delta when you need:
/// - Fast generation (2-8 steps)
/// - Good quality without guidance overhead
/// - Resource-efficient deployment
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT-XL/2 (same as PixArt-Alpha)
/// - Distillation: Latent Consistency Model (LCM)
/// - Steps: 2-8 (optimal: 4)
/// - Guidance scale: 1.0 (no CFG)
/// - Text encoder: T5-XXL (4096-dim)
/// - Resolution: 512x512 to 1024x1024
///
/// Reference: Chen et al., "PixArt-delta: Fast and Controllable Image Generation
/// with Latent Consistency Models", 2024
/// </para>
/// </remarks>
public class PixArtDeltaModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 4096;
    private const double DEFAULT_GUIDANCE_SCALE = 1.0;

    #endregion

    #region Fields

    private DiTNoisePredictor<T> _dit;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PixArtDeltaModel with full customization support.
    /// </summary>
    public PixArtDeltaModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(dit, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        StandardVAE<T>? vae,
        int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 1152,
            numLayers: 28,
            numHeads: 16,
            patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 4,
        double? guidanceScale = null,
        int? seed = null)
    {
        return base.GenerateFromText(
            prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
            combined[i] = ditParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[ditParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var ditCount = _dit.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != ditCount + vaeCount)
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
            ditParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[ditCount + i];

        _dit.SetParameters(ditParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: 1152,
            numLayers: 28, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new PixArtDeltaModel<T>(
            dit: clonedDit, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "PixArt-Delta",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "LCM-distilled PixArt for fast 2-8 step text-to-image generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-xl-2-lcm");
        metadata.SetProperty("distillation_method", "LCM");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("optimal_steps", 4);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE_SCALE);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
