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
/// Lumina Image 2.0 model for high-resolution text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lumina Image 2.0 uses a Flag-DiT (Flow-Aware Generative DiT) architecture with improved
/// multi-resolution support and efficient attention mechanisms. It features Gemma text encoding
/// and flow matching training for generating images up to 2K resolution with excellent detail.
/// </para>
/// <para>
/// <b>For Beginners:</b> Lumina Image 2.0 is an open-source model from the Lumina framework.
///
/// How Lumina Image 2.0 works:
/// 1. Text is encoded by Gemma 2B for strong multilingual understanding
/// 2. A Flag-DiT (Flow-Aware Generative DiT) processes tokens with flow matching
/// 3. Multi-resolution support handles different aspect ratios natively
/// 4. A 16-channel VAE decodes latents to high-resolution images
///
/// Key characteristics:
/// - Flag-DiT architecture with flow-aware generation
/// - Gemma 2B text encoder
/// - Native multi-resolution and aspect ratio support
/// - 2B parameters in the transformer
/// - 16 latent channels
/// - Up to 2K resolution generation
///
/// Advantages:
/// - Open-source with permissive license
/// - Native multi-resolution support
/// - Good quality-to-compute ratio
/// - Flexible aspect ratio handling
/// </para>
/// <para>
/// Reference: Gao et al., "Lumina-Image: High-Resolution Image Generation with
/// Flow-Aware Generative Transformers", 2024
/// </para>
/// </remarks>
public class LuminaImage2Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;

    private const int LUMINA_LATENT_CHANNELS = 16;
    private const int LUMINA_HIDDEN_SIZE = 2048;
    private const int LUMINA_NUM_LAYERS = 32;
    private const int LUMINA_CONTEXT_DIM = 2048;
    private const double LUMINA_DEFAULT_GUIDANCE = 4.0;
    private const int LUMINA_DEFAULT_STEPS = 30;

    #endregion

    #region Fields

    private FlagDiTPredictor<T> _predictor;
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
    public override int LatentChannels => LUMINA_LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    public LuminaImage2Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FlagDiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 1.0, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(LUMINA_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(FlagDiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FlagDiTPredictor<T>(seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LUMINA_LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = LUMINA_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? LUMINA_DEFAULT_GUIDANCE, seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage, string prompt, string? negativePrompt = null,
        double strength = 0.75, int numInferenceSteps = LUMINA_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.ImageToImage(inputImage, prompt, negativePrompt, strength,
            numInferenceSteps, guidanceScale ?? LUMINA_DEFAULT_GUIDANCE, seed);
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
        var cp = new FlagDiTPredictor<T>();
        cp.SetParameters(_predictor.GetParameters());
        var cv = new StandardVAE<T>(inputChannels: 3, latentChannels: LUMINA_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2);
        cv.SetParameters(_vae.GetParameters());
        return new LuminaImage2Model<T>(predictor: cp, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Lumina Image 2.0", Version = "2.0", ModelType = ModelType.NeuralNetwork,
            Description = "Flag-DiT with flow matching, Gemma text encoder, and multi-resolution support",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flag-dit-flow-matching");
        m.SetProperty("text_encoder", "Gemma 2B");
        m.SetProperty("context_dim", LUMINA_CONTEXT_DIM);
        m.SetProperty("hidden_size", LUMINA_HIDDEN_SIZE);
        m.SetProperty("num_layers", LUMINA_NUM_LAYERS);
        m.SetProperty("latent_channels", LUMINA_LATENT_CHANNELS);
        m.SetProperty("multi_resolution", true);
        m.SetProperty("default_resolution", DefaultWidth);
        m.SetProperty("max_resolution", 2048);
        m.SetProperty("default_guidance_scale", LUMINA_DEFAULT_GUIDANCE);
        m.SetProperty("default_inference_steps", LUMINA_DEFAULT_STEPS);
        return m;
    }

    #endregion
}
