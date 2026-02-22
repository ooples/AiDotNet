using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// AuraFlow model — open-source flow-matching text-to-image model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AuraFlow is an open-source flow-matching model from Fal.ai that uses a modified
/// DiT architecture with flow matching for high-quality text-to-image generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> AuraFlow is a community-driven alternative to commercial models:
///
/// Key characteristics:
/// - Flow matching: continuous-time formulation (not discrete DDPM steps)
/// - DiT backbone: transformer-based denoiser
/// - Open-source: fully open weights and code
/// - T5 text encoder for strong prompt understanding
/// - Competitive quality with commercial models
///
/// Use AuraFlow when you need:
/// - Open-source flow-matching model
/// - Alternative to SD3/Flux architecture
/// - Research-friendly model
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT with flow matching
/// - Text encoder: T5-XXL (4096-dim)
/// - Resolution: 1024x1024
/// - Latent channels: 4, 8x downsampling
/// - Scheduler: Flow matching (Euler/midpoint)
///
/// Reference: Fal.ai, "AuraFlow v0.3", 2024
/// </para>
/// </remarks>
public class AuraFlowModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 4096;
    private const double DEFAULT_GUIDANCE_SCALE = 3.5;

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

    public AuraFlowModel(
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
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(dit, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(DiTNoisePredictor<T>? dit, StandardVAE<T>? vae, int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: 1536,
            numLayers: 24, numHeads: 24, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = 25, double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters();
        var vp = _vae.GetParameters();
        var c = new Vector<T>(dp.Length + vp.Length);
        for (int i = 0; i < dp.Length; i++) c[i] = dp[i];
        for (int i = 0; i < vp.Length; i++) c[dp.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int dc = _dit.ParameterCount, vc = _vae.ParameterCount;
        if (parameters.Length != dc + vc)
            throw new ArgumentException($"Expected {dc + vc} parameters, got {parameters.Length}.", nameof(parameters));
        var dp = new Vector<T>(dc);
        var vp = new Vector<T>(vc);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[dc + i];
        _dit.SetParameters(dp);
        _vae.SetParameters(vp);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cd = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: 1536,
            numLayers: 24, numHeads: 24, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        cd.SetParameters(_dit.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        cv.SetParameters(_vae.GetParameters());
        return new AuraFlowModel<T>(dit: cd, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "AuraFlow", Version = "0.3", ModelType = ModelType.NeuralNetwork,
            Description = "AuraFlow open-source flow-matching text-to-image model",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "dit-flow-matching");
        m.SetProperty("text_encoder", "T5-XXL");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }

    #endregion
}
