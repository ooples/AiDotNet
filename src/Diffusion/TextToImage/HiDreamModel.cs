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
/// HiDream-I1 model for high-quality imaginative text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HiDream-I1 uses an MMDiT architecture enhanced with Llama-3.1 as a text encoder,
/// providing superior prompt understanding through a large language model backbone.
/// It features improved composition handling and artistic style diversity compared to
/// standard diffusion models, with variants ranging from 8B to 17B parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> HiDream is designed to generate more creative and imaginative
/// images with excellent prompt understanding.
///
/// How HiDream works:
/// 1. Text is encoded by CLIP ViT-L/14 and Llama-3.1 (a powerful language model)
/// 2. An MMDiT-X transformer processes text and image tokens with joint attention
/// 3. Flow matching training enables efficient 28-50 step generation
/// 4. A 16-channel VAE decodes latents to high-resolution images
///
/// Model variants:
/// - HiDream-I1 Full: 17B parameters, highest quality
/// - HiDream-I1 Dev: 12B parameters, good balance of quality and speed
/// - HiDream-I1 Fast: 8B parameters, optimized for speed
///
/// Key characteristics:
/// - Llama-3.1 as text encoder for deep language understanding
/// - MMDiT-X architecture with enhanced cross-attention
/// - 16 latent channels
/// - Flow matching training
/// - Excellent at artistic and fantasy-style content
/// - Strong composition and spatial reasoning
///
/// Advantages:
/// - Superior prompt understanding via Llama-3.1
/// - Great artistic and creative generation
/// - Multiple speed/quality tradeoff variants
/// - Open-weight model
///
/// Limitations:
/// - Full variant requires significant VRAM (~40GB)
/// - Newer model with smaller community
/// - Llama-3.1 encoder adds overhead
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: MMDiT-X with Llama-3.1 text conditioning
/// - Full: 17B params, hidden 4096, 40 layers
/// - Dev: 12B params, hidden 3072, 32 layers
/// - Fast: 8B params, hidden 2560, 24 layers
/// - Text encoder 1: CLIP ViT-L/14 (768-dim, pooled)
/// - Text encoder 2: Llama-3.1-8B (4096-dim, sequence)
/// - Patch size: 2
/// - VAE: 16 latent channels, 8x spatial compression
/// - Training: Flow matching
/// - Resolution: 1024x1024 default
///
/// Reference: HiDream.ai, "HiDream-I1: High-Quality Imaginative Image Generation", 2025
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create HiDream Full (highest quality)
/// var hiDream = new HiDreamModel&lt;float&gt;(variant: HiDreamVariant.Full);
///
/// // Create HiDream Fast for quicker generation
/// var hiDreamFast = new HiDreamModel&lt;float&gt;(variant: HiDreamVariant.Fast);
///
/// // Generate an image
/// var image = hiDream.GenerateFromText(
///     prompt: "A mystical forest with bioluminescent trees and floating lanterns",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 5.0,
///     seed: 42);
/// </code>
/// </example>
public class HiDreamModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for HiDream (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for HiDream (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int HIDREAM_LATENT_CHANNELS = 16;
    private const int HIDREAM_FULL_HIDDEN_SIZE = 4096;
    private const int HIDREAM_DEV_HIDDEN_SIZE = 3072;
    private const int HIDREAM_FAST_HIDDEN_SIZE = 2560;
    private const int HIDREAM_CONTEXT_DIM = 4096;
    private const double HIDREAM_DEFAULT_GUIDANCE = 5.0;
    private const int HIDREAM_DEFAULT_STEPS = 50;
    private const int HIDREAM_FAST_STEPS = 28;

    #endregion

    #region Fields

    private MMDiTXNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly HiDreamVariant _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => HIDREAM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant.
    /// </summary>
    public HiDreamVariant Variant => _variant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of HiDreamModel.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses HiDream flow matching defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom MMDiT-X noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Dual encoder conditioning (CLIP + Llama-3.1).</param>
    /// <param name="variant">Model variant: Full (17B), Dev (12B), or Fast (8B). Default: Dev.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public HiDreamModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTXNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        HiDreamVariant variant = HiDreamVariant.Dev,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 1.0,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        _variant = variant;

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(HIDREAM_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        MMDiTXNoisePredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Use Large variant for full/dev, Medium for fast
        var mmditVariant = _variant == HiDreamVariant.Fast ? MMDiTXVariant.Medium : MMDiTXVariant.Large;
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(
            variant: mmditVariant,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: HIDREAM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
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
        int numInferenceSteps = HIDREAM_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Fast variant uses fewer default steps
        var effectiveSteps = _variant == HiDreamVariant.Fast && numInferenceSteps == HIDREAM_DEFAULT_STEPS
            ? HIDREAM_FAST_STEPS
            : numInferenceSteps;

        var effectiveGuidanceScale = guidanceScale ?? HIDREAM_DEFAULT_GUIDANCE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            effectiveSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.75,
        int numInferenceSteps = HIDREAM_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? HIDREAM_DEFAULT_GUIDANCE;

        return base.ImageToImage(
            inputImage,
            prompt,
            negativePrompt,
            strength,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var predictorParams = _predictor.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = predictorParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < predictorParams.Length; i++)
            combined[i] = predictorParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[predictorParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var predictorCount = _predictor.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != predictorCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {predictorCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var predictorParams = new Vector<T>(predictorCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < predictorCount; i++)
            predictorParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[predictorCount + i];

        _predictor.SetParameters(predictorParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var mmditVariant = _variant == HiDreamVariant.Fast ? MMDiTXVariant.Medium : MMDiTXVariant.Large;
        var clonedPredictor = new MMDiTXNoisePredictor<T>(variant: mmditVariant);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: HIDREAM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2);
        clonedVae.SetParameters(_vae.GetParameters());

        return new HiDreamModel<T>(
            predictor: clonedPredictor,
            vae: clonedVae,
            conditioner: _conditioner,
            variant: _variant);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        int hiddenSize = _variant switch
        {
            HiDreamVariant.Full => HIDREAM_FULL_HIDDEN_SIZE,
            HiDreamVariant.Fast => HIDREAM_FAST_HIDDEN_SIZE,
            _ => HIDREAM_DEV_HIDDEN_SIZE
        };

        var metadata = new ModelMetadata<T>
        {
            Name = $"HiDream-I1 [{_variant.ToString().ToLowerInvariant()}]",
            Version = _variant.ToString(),
            ModelType = ModelType.NeuralNetwork,
            Description = $"HiDream-I1 [{_variant.ToString().ToLowerInvariant()}] MMDiT-X with Llama-3.1 text encoder for imaginative generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "mmdit-x-llama-conditioning");
        metadata.SetProperty("base_model", "HiDream");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "Llama-3.1-8B");
        metadata.SetProperty("context_dim", HIDREAM_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", hiddenSize);
        metadata.SetProperty("latent_channels", HIDREAM_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("default_guidance_scale", HIDREAM_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", _variant == HiDreamVariant.Fast ? HIDREAM_FAST_STEPS : HIDREAM_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
