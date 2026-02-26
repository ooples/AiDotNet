using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Imagen model for cascaded text-to-image generation with T5 text encoding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Imagen is a cascaded text-to-image diffusion model developed by Google Brain.
/// It demonstrates that large frozen language models (T5-XXL) are highly effective for
/// text-to-image generation, and that scaling the text encoder matters more than scaling the image model.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagen generates images by starting small and upscaling:
///
/// How Imagen works:
/// 1. Text is encoded by a frozen T5-XXL language model (4096-dim embeddings)
/// 2. A base diffusion model generates a 64x64 image from the text embeddings
/// 3. A first super-resolution model upscales 64x64 → 256x256
/// 4. A second super-resolution model upscales 256x256 → 1024x1024
///
/// Key characteristics:
/// - Cascaded pixel-space diffusion: 64→256→1024
/// - Text encoder: Frozen T5-XXL (4.7B parameters, 4096-dim)
/// - Base model: ~2B parameters, 64x64 output
/// - Super-res 1: ~600M parameters, 256x256 output
/// - Super-res 2: ~400M parameters, 1024x1024 output
/// - Uses Efficient U-Net architecture
/// - Dynamic thresholding for improved image quality
///
/// Key innovations:
/// - Demonstrated that text encoder quality is most important
/// - Introduced dynamic thresholding (better high guidance scales)
/// - Efficient U-Net: memory-efficient attention, shifted convolutions
/// - Noise conditioning augmentation for super-resolution stages
///
/// Limitations:
/// - Not open-source (proprietary to Google)
/// - Very large compute requirements
/// - Three separate models needed for full pipeline
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Cascaded pixel-space diffusion with Efficient U-Net
/// - Base model: 64x64, ~2B parameters, 3 RGB channels
/// - Super-res 1: 64→256, ~600M parameters
/// - Super-res 2: 256→1024, ~400M parameters
/// - Text encoder: Frozen T5-XXL (4.7B params, 4096-dim, 256 max tokens)
/// - Noise schedule: Cosine schedule, 1000 training timesteps
/// - Prediction type: Epsilon prediction with dynamic thresholding
/// - Noise conditioning augmentation for super-res stages
///
/// Reference: Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", NeurIPS 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var imagen = new ImagenModel&lt;float&gt;();
///
/// // Generate a 64x64 base image
/// var image = imagen.GenerateFromText(
///     prompt: "A brain riding a rocketship heading towards the moon",
///     negativePrompt: null,
///     width: 64,
///     height: 64,
///     numInferenceSteps: 100,
///     guidanceScale: 7.5,
///     seed: 42);
/// </code>
/// </example>
public class ImagenModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Imagen base model (64x64).
    /// </summary>
    public const int DefaultWidth = 64;

    /// <summary>
    /// Default image height for Imagen base model (64x64).
    /// </summary>
    public const int DefaultHeight = 64;

    private const int IMAGEN_PIXEL_CHANNELS = 3;

    /// <summary>
    /// Cross-attention dimension matching T5-XXL output (4096).
    /// </summary>
    private const int IMAGEN_CROSS_ATTENTION_DIM = 4096;

    /// <summary>
    /// Default guidance scale for Imagen (7.5).
    /// </summary>
    private const double IMAGEN_DEFAULT_GUIDANCE_SCALE = 7.5;

    /// <summary>
    /// Default dynamic thresholding percentile (99.5%).
    /// </summary>
    private const double IMAGEN_DYNAMIC_THRESHOLD_PERCENTILE = 0.995;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _baseUnet;
    private UNetNoisePredictor<T> _superRes1Unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly double _dynamicThresholdPercentile;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => IMAGEN_PIXEL_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _baseUnet.ParameterCount + _superRes1Unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the super-resolution Stage 1 noise predictor (64→256).
    /// </summary>
    public INoisePredictor<T> SuperResolution1 => _superRes1Unet;

    /// <summary>
    /// Gets the cross-attention dimension (4096 for T5-XXL).
    /// </summary>
    public int CrossAttentionDim => IMAGEN_CROSS_ATTENTION_DIM;

    /// <summary>
    /// Gets the dynamic thresholding percentile.
    /// </summary>
    public double DynamicThresholdPercentile => _dynamicThresholdPercentile;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of ImagenModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses Imagen defaults: cosine beta schedule, 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with Imagen settings.
    /// </param>
    /// <param name="baseUnet">
    /// Custom base U-Net (64x64 generation). If null, creates the standard ~2B parameter Efficient U-Net.
    /// </param>
    /// <param name="superRes1Unet">
    /// Custom super-resolution U-Net (64→256). If null, creates the standard ~600M parameter model.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates a minimal pixel-space VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically T5-XXL).
    /// </param>
    /// <param name="dynamicThresholdPercentile">
    /// Percentile for dynamic thresholding (default: 0.995 = 99.5th percentile).
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ImagenModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUnet = null,
        UNetNoisePredictor<T>? superRes1Unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        double dynamicThresholdPercentile = IMAGEN_DYNAMIC_THRESHOLD_PERCENTILE,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.SquaredCosine
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _dynamicThresholdPercentile = dynamicThresholdPercentile;

        InitializeLayers(baseUnet, superRes1Unet, vae, seed);

        SetGuidanceScale(IMAGEN_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Efficient U-Net base model, super-resolution model, and VAE,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the Imagen paper.
    /// </summary>
    [MemberNotNull(nameof(_baseUnet), nameof(_superRes1Unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? baseUnet,
        UNetNoisePredictor<T>? superRes1Unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Base model: Efficient U-Net (~2B parameters)
        // Generates 64x64 RGB images from T5-XXL text embeddings
        _baseUnet = baseUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: IMAGEN_PIXEL_CHANNELS,
            outputChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 4, 8],
            numResBlocks: 3,
            attentionResolutions: [8, 4, 2],
            contextDim: IMAGEN_CROSS_ATTENTION_DIM,
            seed: seed);

        // Super-resolution 1: Efficient U-Net (~600M parameters)
        // Upscales 64x64 → 256x256 with noise conditioning augmentation
        _superRes1Unet = superRes1Unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: IMAGEN_PIXEL_CHANNELS * 2,
            outputChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: IMAGEN_CROSS_ATTENTION_DIM,
            seed: seed);

        // Minimal pixel-space VAE (identity-like)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 1.0,
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
        int numInferenceSteps = 100,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IMAGEN_DEFAULT_GUIDANCE_SCALE;

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
        int numInferenceSteps = 100,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IMAGEN_DEFAULT_GUIDANCE_SCALE;

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
        var baseParams = _baseUnet.GetParameters();
        var sr1Params = _superRes1Unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = baseParams.Length + sr1Params.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        var offset = 0;
        for (int i = 0; i < baseParams.Length; i++)
        {
            combined[offset + i] = baseParams[i];
        }
        offset += baseParams.Length;

        for (int i = 0; i < sr1Params.Length; i++)
        {
            combined[offset + i] = sr1Params[i];
        }
        offset += sr1Params.Length;

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var baseCount = _baseUnet.ParameterCount;
        var sr1Count = _superRes1Unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != baseCount + sr1Count + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {baseCount + sr1Count + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var baseParams = new Vector<T>(baseCount);
        var sr1Params = new Vector<T>(sr1Count);
        var vaeParams = new Vector<T>(vaeCount);

        var offset = 0;
        for (int i = 0; i < baseCount; i++)
        {
            baseParams[i] = parameters[offset + i];
        }
        offset += baseCount;

        for (int i = 0; i < sr1Count; i++)
        {
            sr1Params[i] = parameters[offset + i];
        }
        offset += sr1Count;

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset + i];
        }

        _baseUnet.SetParameters(baseParams);
        _superRes1Unet.SetParameters(sr1Params);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedBase = new UNetNoisePredictor<T>(
            inputChannels: IMAGEN_PIXEL_CHANNELS,
            outputChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 4, 8],
            numResBlocks: 3,
            attentionResolutions: [8, 4, 2],
            contextDim: IMAGEN_CROSS_ATTENTION_DIM);
        clonedBase.SetParameters(_baseUnet.GetParameters());

        var clonedSR1 = new UNetNoisePredictor<T>(
            inputChannels: IMAGEN_PIXEL_CHANNELS * 2,
            outputChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: IMAGEN_CROSS_ATTENTION_DIM);
        clonedSR1.SetParameters(_superRes1Unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IMAGEN_PIXEL_CHANNELS,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 1.0);
        clonedVae.SetParameters(_vae.GetParameters());

        return new ImagenModel<T>(
            baseUnet: clonedBase,
            superRes1Unet: clonedSR1,
            vae: clonedVae,
            conditioner: _conditioner,
            dynamicThresholdPercentile: _dynamicThresholdPercentile);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Imagen",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Imagen cascaded pixel-space diffusion model with T5-XXL text encoder and dynamic thresholding",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-pixel-diffusion");
        metadata.SetProperty("base_model", "Imagen");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("cross_attention_dim", IMAGEN_CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_resolution", 64);
        metadata.SetProperty("sr1_resolution", 256);
        metadata.SetProperty("sr2_resolution", 1024);
        metadata.SetProperty("pixel_channels", IMAGEN_PIXEL_CHANNELS);
        metadata.SetProperty("dynamic_threshold_percentile", _dynamicThresholdPercentile);

        return metadata;
    }

    #endregion
}
