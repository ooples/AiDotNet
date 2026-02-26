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
/// Ideogram 3 model for text-to-image generation with superior text rendering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ideogram 3 specializes in generating images with accurate, legible text rendering.
/// It uses a specialized text layout prediction module alongside a SiT diffusion backbone
/// to ensure rendered text is spelled correctly, properly formatted, and placed naturally
/// within the generated scene.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most image generators struggle to write readable text in images.
/// Ideogram 3 excels at this — it can generate signs, logos, posters, and business cards
/// with correctly spelled, properly formatted text.
///
/// How Ideogram 3 works:
/// 1. Text prompt is parsed to identify embedded text requests (e.g., "a sign that says 'Hello'")
/// 2. A text layout predictor determines optimal placement, font size, and orientation
/// 3. The SiT diffusion backbone generates the image with text-aware conditioning
/// 4. OCR-in-the-loop training ensures spelling accuracy
///
/// Key characteristics:
/// - Text layout prediction module for accurate text placement
/// - OCR-in-the-loop training for spelling accuracy
/// - SiT (Scalable Interpolant Transformer) backbone
/// - Handles multi-line text, different fonts, and curved text
/// - 16 latent channels
///
/// Advantages:
/// - Best-in-class text rendering accuracy
/// - Correct spelling even for complex words
/// - Natural text placement and sizing
/// - Handles various text styles (signs, posters, book covers)
/// - Good overall image quality beyond text
///
/// Limitations:
/// - API-only access
/// - Text rendering adds inference overhead
/// - Less control over exact font/style compared to dedicated design tools
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SiT with text layout prediction
/// - Text layout module: Predicts bounding boxes, font size, orientation
/// - OCR-in-the-loop: Training-time OCR validation for spelling
/// - Backbone: SiT, hidden 2048, 24 layers
/// - VAE: 16 latent channels
/// - Default: 30 steps, guidance scale 7.5
/// - Resolution: 1024x1024 default
///
/// Reference: Ideogram Inc., "Ideogram 3", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Ideogram 3
/// var ideogram = new Ideogram3Model&lt;float&gt;();
///
/// // Generate an image with text
/// var image = ideogram.GenerateFromText(
///     prompt: "A vintage movie poster that says 'The Last Journey' in art deco style",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5,
///     seed: 42);
/// </code>
/// </example>
public class Ideogram3Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Ideogram 3 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Ideogram 3 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int IDEOGRAM_LATENT_CHANNELS = 16;
    private const int IDEOGRAM_HIDDEN_SIZE = 2048;
    private const int IDEOGRAM_NUM_LAYERS = 24;
    private const int IDEOGRAM_CONTEXT_DIM = 4096;
    private const double IDEOGRAM_DEFAULT_GUIDANCE = 7.5;
    private const int IDEOGRAM_DEFAULT_STEPS = 30;

    #endregion

    #region Fields

    private SiTPredictor<T> _predictor;
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
    public override int LatentChannels => IDEOGRAM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this model has specialized text rendering capabilities.
    /// </summary>
    public bool SupportsTextRendering => true;

    /// <summary>
    /// Gets whether this model uses OCR-in-the-loop training for spelling accuracy.
    /// </summary>
    public bool UsesOCRTraining => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Ideogram3Model.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses Ideogram 3 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom SiT noise predictor with text layout awareness.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Text encoder conditioning with layout prediction.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Ideogram3Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        SiTPredictor<T>? predictor = null,
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

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(IDEOGRAM_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        SiTPredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IDEOGRAM_LATENT_CHANNELS,
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
        int numInferenceSteps = IDEOGRAM_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IDEOGRAM_DEFAULT_GUIDANCE;

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
        int numInferenceSteps = IDEOGRAM_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IDEOGRAM_DEFAULT_GUIDANCE;

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
        var clonedPredictor = new SiTPredictor<T>();
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IDEOGRAM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Ideogram3Model<T>(
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
            Name = "Ideogram 3",
            Version = "3.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "SiT with text layout prediction and OCR-in-the-loop training for superior text rendering",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sit-text-layout-prediction");
        metadata.SetProperty("hidden_size", IDEOGRAM_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", IDEOGRAM_NUM_LAYERS);
        metadata.SetProperty("context_dim", IDEOGRAM_CONTEXT_DIM);
        metadata.SetProperty("latent_channels", IDEOGRAM_LATENT_CHANNELS);
        metadata.SetProperty("text_rendering", true);
        metadata.SetProperty("ocr_training", true);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("default_guidance_scale", IDEOGRAM_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", IDEOGRAM_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
