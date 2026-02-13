using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Video;

/// <summary>
/// Lumina-T2X model — transformer-based text-to-any generation (image, video, 3D, audio).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lumina-T2X is a unified transformer framework for generating multiple modalities
/// from text: images, videos, 3D objects, and audio. It uses a Flag-DiT backbone
/// with resolution-aware positional encoding.
/// </para>
/// <para>
/// <b>For Beginners:</b> Lumina-T2X is a unified model for multiple generation tasks:
///
/// Key characteristics:
/// - Single architecture generates images, videos, 3D, and audio
/// - Flag-DiT: improved DiT with flow matching
/// - Resolution-aware encoding: handles any aspect ratio
/// - Gemma text encoder for multilingual prompts
/// - Scalable from 0.6B to 7B parameters
///
/// Use Lumina-T2X when you need:
/// - Multi-modal generation from text
/// - Flexible resolution/aspect ratio support
/// - Research into unified generation
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Flag-DiT with flow matching
/// - Text encoder: Gemma-2B (2048-dim)
/// - Resolution: up to 2048x2048
/// - Latent channels: 4, 8x downsampling
/// - Flow matching with velocity prediction
///
/// Reference: Gao et al., "Lumina-T2X: Transforming Text into Any Modality", 2024
/// </para>
/// </remarks>
public class LuminaT2XModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 2048;
    private const double DEFAULT_GUIDANCE_SCALE = 4.0;

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

    public LuminaT2XModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()))
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
            numLayers: 30, numHeads: 16, patchSize: 2,
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
        int numInferenceSteps = 30, double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters(); var vp = _vae.GetParameters();
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
            throw new ArgumentException($"Expected {dc + vc}, got {parameters.Length}.", nameof(parameters));
        var dp = new Vector<T>(dc); var vp = new Vector<T>(vc);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[dc + i];
        _dit.SetParameters(dp); _vae.SetParameters(vp);
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
            numLayers: 30, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        cd.SetParameters(_dit.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        cv.SetParameters(_vae.GetParameters());
        return new LuminaT2XModel<T>(dit: cd, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Lumina-T2X", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Lumina-T2X unified text-to-any generation framework",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flag-dit-flow-matching");
        m.SetProperty("modalities", "image,video,3d,audio");
        m.SetProperty("text_encoder", "Gemma-2B");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }

    #endregion
}
