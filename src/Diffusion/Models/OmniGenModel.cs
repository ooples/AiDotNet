using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// OmniGen model — unified image generation model handling multiple tasks in one architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OmniGen is a unified image generation model that handles text-to-image, image editing,
/// subject-driven generation, and visual conditional generation with a single model,
/// without requiring task-specific adapters or fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> OmniGen is one model that does everything:
///
/// Key characteristics:
/// - Unified model: text-to-image, editing, inpainting, subject-driven, etc.
/// - No task-specific adapters needed (unlike ControlNet, IP-Adapter)
/// - Single transformer backbone handles all tasks
/// - Interleaved image-text input for flexible conditioning
/// - In-context learning: understands task from examples
///
/// Tasks OmniGen can handle:
/// - Text-to-image generation
/// - Image editing (instruction-based)
/// - Subject-driven generation (given reference images)
/// - Visual conditional generation (depth, edge, pose)
/// - Style transfer
///
/// Use OmniGen when you need:
/// - Single model for multiple generation tasks
/// - Simplified pipeline (no adapter management)
/// - Flexible conditioning from images and text
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Unified transformer with interleaved image-text tokens
/// - Parameters: ~3.8B
/// - Text encoder: integrated (not separate)
/// - Resolution: 512x512 to 1024x1024
/// - Latent channels: 4, 8x downsampling
///
/// Reference: Xiao et al., "OmniGen: Unified Image Generation", 2024
/// </para>
/// </remarks>
public class OmniGenModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 2048;
    private const double DEFAULT_GUIDANCE_SCALE = 3.0;

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

    public OmniGenModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()))
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
            inputChannels: LATENT_CHANNELS, hiddenSize: 2048,
            numLayers: 32, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = 50, double? guidanceScale = null, int? seed = null)
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
            inputChannels: LATENT_CHANNELS, hiddenSize: 2048,
            numLayers: 32, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        cd.SetParameters(_dit.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        cv.SetParameters(_vae.GetParameters());
        return new OmniGenModel<T>(dit: cd, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "OmniGen", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "OmniGen unified multi-task image generation model",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "unified-dit");
        m.SetProperty("tasks", "text2img,editing,subject-driven,conditional");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }

    #endregion
}
