using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// MidJourney V7-style model architecture for artistic text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Represents the MidJourney V7-style architecture known for highly artistic and photorealistic
/// generation. Uses a multi-scale MMDiT-X architecture with aesthetic-aware training, enhanced
/// prompt interpretation via a proprietary language model, and a stylize parameter for controlling
/// the artistic vs photorealistic balance.
/// </para>
/// <para>
/// <b>For Beginners:</b> MidJourney V7 is renowned for producing the most visually stunning
/// and artistic images in the AI generation space.
///
/// How MidJourney V7 works:
/// 1. Text is deeply interpreted by a proprietary language model for nuanced understanding
/// 2. A multi-scale MMDiT-X processes tokens at different resolution scales
/// 3. Aesthetic-aware training ensures consistently beautiful output
/// 4. A stylize parameter controls artistic interpretation vs literal prompt following
///
/// Key characteristics:
/// - Multi-scale MMDiT-X for coherent generation across scales
/// - Proprietary language model for deep prompt interpretation
/// - Aesthetic-aware training with human preference data
/// - Stylize parameter (0-1000) controlling artistic expression
/// - Strong photorealistic and artistic capabilities
/// - 16 latent channels
///
/// Advantages:
/// - Best-in-class artistic and aesthetic quality
/// - Exceptional at photorealism and creative composition
/// - Strong prompt interpretation and creative expansion
/// - Stylize parameter gives fine control over output style
///
/// Limitations:
/// - API-only (not open-source, architecture details are proprietary)
/// - Less precise prompt adherence when stylize is high
/// - Requires internet connection and subscription
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Multi-scale MMDiT-X with aesthetic training
/// - Hidden size: estimated ~4096, ~40+ layers
/// - Stylize range: 0 (literal) to 1000 (highly artistic)
/// - VAE: 16 latent channels
/// - Default: 50 steps, guidance scale 5.0
/// - Resolution: 1024x1024 default, aspect-ratio aware
///
/// Note: MidJourney is proprietary; this is a best-effort architectural representation
/// based on publicly available information.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create MidJourney V7
/// var mj7 = new MidJourneyV7Model&lt;float&gt;();
///
/// // Generate an artistic image
/// var image = mj7.GenerateFromText(
///     prompt: "ethereal forest spirit emerging from morning mist, cinematic lighting",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 5.0,
///     seed: 42);
/// </code>
/// </example>
public class MidJourneyV7Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;

    private const int MJ7_LATENT_CHANNELS = 16;
    private const int MJ7_HIDDEN_SIZE = 4096;
    private const int MJ7_NUM_LAYERS = 40;
    private const double MJ7_DEFAULT_GUIDANCE = 5.0;
    private const int MJ7_DEFAULT_STEPS = 50;

    #endregion

    #region Fields

    private MMDiTXNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => MJ7_LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    public MidJourneyV7Model(
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
                TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(MJ7_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(MMDiTXNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(variant: MMDiTXVariant.Large, seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: MJ7_LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = MJ7_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? MJ7_DEFAULT_GUIDANCE, seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage, string prompt, string? negativePrompt = null,
        double strength = 0.75, int numInferenceSteps = MJ7_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.ImageToImage(inputImage, prompt, negativePrompt, strength,
            numInferenceSteps, guidanceScale ?? MJ7_DEFAULT_GUIDANCE, seed);
    }

    #endregion

    #region IParameterizable Implementation

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

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cp = new MMDiTXNoisePredictor<T>(variant: MMDiTXVariant.Large);
        cp.SetParameters(_predictor.GetParameters());
        var cv = new StandardVAE<T>(inputChannels: 3, latentChannels: MJ7_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2);
        cv.SetParameters(_vae.GetParameters());
        return new MidJourneyV7Model<T>(predictor: cp, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "MidJourney V7", Version = "7.0", ModelType = ModelType.NeuralNetwork,
            Description = "Multi-scale MMDiT-X with aesthetic-aware training and stylize control",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "multi-scale-mmdit-x-aesthetic");
        m.SetProperty("hidden_size", MJ7_HIDDEN_SIZE);
        m.SetProperty("num_layers", MJ7_NUM_LAYERS);
        m.SetProperty("latent_channels", MJ7_LATENT_CHANNELS);
        m.SetProperty("stylize_control", true);
        m.SetProperty("default_resolution", DefaultWidth);
        m.SetProperty("default_guidance_scale", MJ7_DEFAULT_GUIDANCE);
        m.SetProperty("default_inference_steps", MJ7_DEFAULT_STEPS);
        return m;
    }

    #endregion
}
