using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Imagen 2 model for improved cascaded text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Imagen 2 is Google DeepMind's improved text-to-image model, successor to the original Imagen.
/// It features improved image quality, better text rendering, and enhanced prompt following.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagen 2 is Google's upgraded image generator:
///
/// Key improvements over Imagen 1:
/// - Much better text rendering in images
/// - Improved photorealism and detail
/// - Better safety filtering
/// - Enhanced prompt following
///
/// How Imagen 2 works:
/// 1. Text encoder produces rich embeddings
/// 2. Base model generates low-resolution image (64x64)
/// 3. Cascaded super-resolution: 64 -> 256 -> 1024
/// 4. Each stage uses U-Net with cross-attention to text
///
/// Use Imagen 2 when you need:
/// - Photorealistic image generation
/// - Text rendered correctly in images
/// - High fidelity to complex prompts
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Cascaded latent diffusion with Efficient U-Net
/// - Base model: 64x64 generation
/// - Super-resolution stages: 64->256, 256->1024
/// - Text encoder: T5-XXL (frozen, 4096-dim)
/// - Dynamic thresholding for high guidance scales
///
/// Reference: Google DeepMind, "Imagen 2", 2023
/// </para>
/// </remarks>
public class Imagen2Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 4096;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _isImagen3;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this is the Imagen 3 variant.
    /// </summary>
    public bool IsImagen3 => _isImagen3;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Imagen2Model with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses Imagen 2 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="unet">Custom U-Net noise predictor.</param>
    /// <param name="vae">Custom VAE.</param>
    /// <param name="conditioner">Text encoder conditioning module.</param>
    /// <param name="isImagen3">Whether to use Imagen 3 configuration (default: false).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Imagen2Model(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        bool isImagen3 = false,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()))
    {
        _conditioner = conditioner;
        _isImagen3 = isImagen3;
        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        int baseChannels = _isImagen3 ? 384 : 320;
        int contextDim = _isImagen3 ? 4096 : CROSS_ATTENTION_DIM;

        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: baseChannels,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: contextDim,
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
        int numInferenceSteps = 50,
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
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);

        for (int i = 0; i < unetParams.Length; i++)
            combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[unetParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[unetCount + i];

        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        int baseChannels = _isImagen3 ? 384 : 320;
        int contextDim = _isImagen3 ? 4096 : CROSS_ATTENTION_DIM;

        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: baseChannels, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3, attentionResolutions: [4, 2, 1],
            contextDim: contextDim);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Imagen2Model<T>(
            unet: clonedUnet, vae: clonedVae,
            conditioner: _conditioner, isImagen3: _isImagen3);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var name = _isImagen3 ? "Imagen 3" : "Imagen 2";
        var metadata = new ModelMetadata<T>
        {
            Name = name,
            Version = _isImagen3 ? "3.0" : "2.0",
            ModelType = ModelType.NeuralNetwork,
            Description = $"{name} cascaded text-to-image generation with T5-XXL text encoding",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-latent-diffusion");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("is_imagen3", _isImagen3);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates an Imagen 3 model instance.
    /// </summary>
    public static Imagen2Model<T> CreateImagen3(int? seed = null)
    {
        return new Imagen2Model<T>(isImagen3: true, seed: seed);
    }

    #endregion
}
