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
/// Playground v3 model for aesthetically optimized text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Playground v3 is optimized for aesthetic quality through human preference training with
/// DPO (Direct Preference Optimization). It uses an MMDiT-X architecture similar to SD3.5
/// but fine-tuned extensively on human aesthetic preference data, producing images that
/// consistently score highly on visual appeal metrics.
/// </para>
/// <para>
/// <b>For Beginners:</b> Playground v3 is specifically trained to generate beautiful,
/// aesthetically pleasing images.
///
/// How Playground v3 works:
/// 1. Text is encoded by triple encoders: CLIP ViT-L, OpenCLIP ViT-bigG, and T5-XXL
/// 2. An MMDiT-X transformer (same family as SD3.5) denoises the latent
/// 3. Aesthetic reward training ensures outputs are visually appealing
/// 4. EDM2-style preconditioning improves sampling efficiency
///
/// Key characteristics:
/// - MMDiT-X architecture optimized for aesthetics
/// - DPO (Direct Preference Optimization) training for visual appeal
/// - Triple text encoders for comprehensive prompt understanding
/// - EDM2-style preconditioning
/// - 16 latent channels
/// - 25 default inference steps
///
/// Advantages:
/// - Consistently generates aesthetically beautiful images
/// - Strong color harmony and composition
/// - Excellent skin tones and lighting in portraits
/// - Good balance of realism and artistic quality
///
/// Limitations:
/// - May sacrifice strict prompt adherence for aesthetics
/// - Aesthetic bias may not suit all use cases
/// - Limited stylistic diversity compared to unaligned models
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: MMDiT-X with aesthetic DPO alignment
/// - Backbone: ~8B params, hidden 4096, 38 layers, 64 heads
/// - Text encoder 1: CLIP ViT-L/14 (768-dim)
/// - Text encoder 2: OpenCLIP ViT-bigG (1280-dim)
/// - Text encoder 3: T5-XXL (4096-dim)
/// - Training: Rectified flow + DPO aesthetic alignment
/// - VAE: 16 latent channels, 8x spatial compression
/// - Default: 25 steps, guidance scale 5.0
/// - Resolution: 1024x1024 default
///
/// Reference: Li et al., "Playground v3: Improving Text-to-Image Alignment with
/// Human Feedback", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Playground v3
/// var playground = new PlaygroundV3Model&lt;float&gt;();
///
/// // Generate an aesthetically pleasing image
/// var image = playground.GenerateFromText(
///     prompt: "A beautiful portrait of a woman in soft golden hour light, bokeh background",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 25,
///     guidanceScale: 5.0,
///     seed: 42);
/// </code>
/// </example>
public class PlaygroundV3Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Playground v3 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Playground v3 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int PG3_LATENT_CHANNELS = 16;
    private const int PG3_HIDDEN_SIZE = 4096;
    private const int PG3_NUM_LAYERS = 38;
    private const int PG3_NUM_HEADS = 64;
    private const int PG3_CONTEXT_DIM = 4096;
    private const double PG3_DEFAULT_GUIDANCE = 5.0;
    private const int PG3_DEFAULT_STEPS = 25;

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
    public override int LatentChannels => PG3_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this model uses aesthetic DPO alignment.
    /// </summary>
    public bool UsesAestheticDPO => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PlaygroundV3Model.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses Playground v3 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom MMDiT-X noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Triple text encoder conditioning.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PlaygroundV3Model(
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
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(PG3_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        MMDiTXNoisePredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(
            variant: MMDiTXVariant.Large,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: PG3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305,
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
        int numInferenceSteps = PG3_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? PG3_DEFAULT_GUIDANCE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.75,
        int numInferenceSteps = PG3_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? PG3_DEFAULT_GUIDANCE;

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
        var clonedPredictor = new MMDiTXNoisePredictor<T>(variant: MMDiTXVariant.Large);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: PG3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305);
        clonedVae.SetParameters(_vae.GetParameters());

        return new PlaygroundV3Model<T>(
            predictor: clonedPredictor,
            vae: clonedVae,
            conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Playground v3",
            Version = "3.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Aesthetically optimized MMDiT-X with DPO human preference alignment",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "mmdit-x-aesthetic-dpo");
        metadata.SetProperty("base_model", "Playground v3");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "OpenCLIP ViT-bigG");
        metadata.SetProperty("text_encoder_3", "T5-XXL");
        metadata.SetProperty("context_dim", PG3_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", PG3_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", PG3_NUM_LAYERS);
        metadata.SetProperty("num_heads", PG3_NUM_HEADS);
        metadata.SetProperty("latent_channels", PG3_LATENT_CHANNELS);
        metadata.SetProperty("aesthetic_dpo", true);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("default_guidance_scale", PG3_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", PG3_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
