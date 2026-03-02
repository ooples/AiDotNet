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
/// Lumina-T2X unified framework for transforming text into any modality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lumina-T2X is a unified framework for text-to-any generation using Flag-DiT (Flow-Aware
/// Generative DiT) blocks. It supports generating images, videos, 3D content, and audio from
/// text prompts using a shared transformer backbone with modality-specific output heads.
/// This implementation focuses on the image generation modality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Lumina-T2X is a versatile model that can generate not just images,
/// but potentially video, audio, and 3D content too — all from text descriptions.
///
/// How Lumina-T2X works:
/// 1. Text is encoded by Gemma 2B for broad language understanding
/// 2. A shared Flag-DiT backbone processes the conditioning
/// 3. Modality-specific heads produce output for the target format
/// 4. Flow matching enables flexible resolution and duration handling
///
/// Key characteristics:
/// - Unified backbone for multi-modality generation
/// - Flag-DiT architecture with flow matching
/// - Gemma 2B text encoder
/// - Modality-specific output heads (image, video, audio, 3D)
/// - Resolution-agnostic and duration-agnostic design
///
/// Advantages:
/// - Single model backbone for multiple modalities
/// - Open-source and well-documented
/// - Flexible resolution and aspect ratio support
/// - Clean modular design for extension
/// </para>
/// <para>
/// Reference: Gao et al., "Lumina-T2X: Transforming Text into Any Modality,
/// Resolution, and Duration via Flow-based Large Diffusion Transformers", 2024
/// </para>
/// </remarks>
public class LuminaT2XModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;

    private const int T2X_LATENT_CHANNELS = 16;
    private const int T2X_HIDDEN_SIZE = 2048;
    private const int T2X_NUM_LAYERS = 32;
    private const int T2X_CONTEXT_DIM = 2048;
    private const double T2X_DEFAULT_GUIDANCE = 4.0;
    private const int T2X_DEFAULT_STEPS = 30;

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
    public override int LatentChannels => T2X_LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this model supports multi-modality output.
    /// </summary>
    public bool SupportsMultiModality => true;

    #endregion

    #region Constructor

    public LuminaT2XModel(
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
        SetGuidanceScale(T2X_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(FlagDiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FlagDiTPredictor<T>(seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: T2X_LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = T2X_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? T2X_DEFAULT_GUIDANCE, seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage, string prompt, string? negativePrompt = null,
        double strength = 0.75, int numInferenceSteps = T2X_DEFAULT_STEPS,
        double? guidanceScale = null, int? seed = null)
    {
        return base.ImageToImage(inputImage, prompt, negativePrompt, strength,
            numInferenceSteps, guidanceScale ?? T2X_DEFAULT_GUIDANCE, seed);
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
        var cv = new StandardVAE<T>(inputChannels: 3, latentChannels: T2X_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2);
        cv.SetParameters(_vae.GetParameters());
        return new LuminaT2XModel<T>(predictor: cp, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Lumina-T2X", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Unified text-to-any framework with Flag-DiT for multi-modality generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flag-dit-unified-multimodal");
        m.SetProperty("base_model", "Lumina-T2X");
        m.SetProperty("text_encoder", "Gemma 2B");
        m.SetProperty("context_dim", T2X_CONTEXT_DIM);
        m.SetProperty("hidden_size", T2X_HIDDEN_SIZE);
        m.SetProperty("num_layers", T2X_NUM_LAYERS);
        m.SetProperty("latent_channels", T2X_LATENT_CHANNELS);
        m.SetProperty("multi_modality", true);
        m.SetProperty("supported_modalities", "image, video, audio, 3D");
        m.SetProperty("default_resolution", DefaultWidth);
        m.SetProperty("default_guidance_scale", T2X_DEFAULT_GUIDANCE);
        m.SetProperty("default_inference_steps", T2X_DEFAULT_STEPS);
        return m;
    }

    #endregion
}
