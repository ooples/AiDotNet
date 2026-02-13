using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// Null-text Inversion model for editing real images by optimizing unconditional embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Null-text Inversion enables precise editing of real photographs by first inverting them
/// into latent space using DDIM inversion, then optimizing the unconditional (null-text)
/// embedding at each timestep to ensure faithful reconstruction. Once inverted, standard
/// text-guided editing techniques like Prompt-to-Prompt can modify the image.
/// </para>
/// <para>
/// <b>For Beginners:</b> Null-text Inversion lets you edit real photos with text prompts.
///
/// How it works:
/// 1. DDIM inversion maps your real image back to noise (latent code)
/// 2. The null-text embedding is optimized per-timestep for accurate reconstruction
/// 3. Editing is performed by swapping the text prompt while keeping the optimized embeddings
///
/// Key characteristics:
/// - Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
/// - Enables Prompt-to-Prompt editing on real images
/// - Requires per-image optimization (~50 optimization steps per timestep)
/// - Uses DDIM scheduler for deterministic inversion and generation
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net with per-timestep null-text optimization
/// - Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
/// - Cross-attention dimension: 768
/// - VAE: 4 latent channels, scale factor 0.18215
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Scheduler: DDIM (required for deterministic inversion)
///
/// Reference: Mokady et al., "Null-text Inversion for Editing Real Images using Guided Diffusion Models", CVPR 2023
/// </para>
/// </remarks>
public class NullTextInversionModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Null-text Inversion (SD 1.5 native resolution).
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for Null-text Inversion (SD 1.5 native resolution).
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels in the VAE.
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-L/14 output (768).
    /// </summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>
    /// Default classifier-free guidance scale (7.5).
    /// </summary>
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

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
    /// Gets the cross-attention dimension (768 for CLIP ViT-L/14).
    /// </summary>
    public int CrossAttentionDim => CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of NullTextInversionModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses SD 1.5 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, creates a DDIM scheduler.</param>
    /// <param name="unet">Custom U-Net. If null, creates the standard SD 1.5 U-Net.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD 1.5 VAE.</param>
    /// <param name="conditioner">Text encoder conditioning module.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public NullTextInversionModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
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
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers using custom layers if provided,
    /// or creating industry-standard SD 1.5 layers.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM,
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
        var effectiveGuidanceScale = guidanceScale ?? DEFAULT_GUIDANCE_SCALE;

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
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? DEFAULT_GUIDANCE_SCALE;

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
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(unetParams);
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
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new NullTextInversionModel<T>(
            unet: clonedUnet,
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
            Name = "Null-text Inversion",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Null-text Inversion enables editing real images by optimizing the unconditional embedding",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "latent-diffusion");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "DDIM");

        return metadata;
    }

    #endregion
}
