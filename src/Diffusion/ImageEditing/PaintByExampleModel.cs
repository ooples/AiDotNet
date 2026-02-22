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

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// Paint-by-Example model for exemplar-based inpainting using reference images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Paint-by-Example fills masked image regions using exemplar images as visual references
/// instead of text prompts. The U-Net receives 9 input channels: 4 latent channels from the
/// noisy latent, 4 channels from the masked source image latent, and 1 channel for the binary
/// mask. The exemplar image is encoded via CLIP and injected through cross-attention.
/// </para>
/// <para>
/// <b>For Beginners:</b> Paint-by-Example fills in parts of an image using another image as a guide.
///
/// How it works:
/// 1. You provide a source image, a mask indicating the region to fill, and a reference image
/// 2. The reference image is encoded by CLIP into a visual embedding
/// 3. The U-Net takes 9 input channels (latent + masked image + mask) and denoises
/// 4. The filled region matches the style and content of the reference image
///
/// Key characteristics:
/// - Based on Stable Diffusion 1.5 (512x512)
/// - U-Net has 9 input channels (4 latent + 4 masked image + 1 mask)
/// - Uses exemplar images instead of text for conditioning
/// - Uses DDIM scheduler for deterministic inpainting results
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net modified with 9 input channels for inpainting
/// - Conditioning: CLIP image encoder (768-dim) via cross-attention
/// - Input channels: 9 (4 noisy latent + 4 masked source latent + 1 binary mask)
/// - Output channels: 4 (latent space prediction)
/// - Cross-attention dimension: 768
/// - VAE: 4 latent channels, scale factor 0.18215
/// - Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
/// - Scheduler: DDIM
///
/// Reference: Yang et al., "Paint by Example: Exemplar-based Image Editing with Diffusion Models", CVPR 2023
/// </para>
/// </remarks>
public class PaintByExampleModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Paint-by-Example (SD 1.5 native resolution).
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for Paint-by-Example (SD 1.5 native resolution).
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Number of latent channels in the VAE.
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Number of U-Net input channels (4 latent + 4 masked image + 1 mask).
    /// </summary>
    private const int INPUT_CHANNELS = 9;

    /// <summary>
    /// Cross-attention dimension matching CLIP image encoder output (768).
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
    /// Gets the cross-attention dimension (768 for CLIP image encoder).
    /// </summary>
    public int CrossAttentionDim => CROSS_ATTENTION_DIM;

    /// <summary>
    /// Gets the number of U-Net input channels (9: 4 latent + 4 masked + 1 mask).
    /// </summary>
    public int InputChannels => INPUT_CHANNELS;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PaintByExampleModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses SD 1.5 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, creates a DDIM scheduler.</param>
    /// <param name="unet">Custom U-Net. If null, creates a 9-channel input U-Net for inpainting.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SD 1.5 VAE.</param>
    /// <param name="conditioner">Image encoder conditioning module (typically CLIP image encoder).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PaintByExampleModel(
        NeuralNetworkArchitecture<T>? architecture = null,
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
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers using custom layers if provided,
    /// or creating industry-standard layers for exemplar-based inpainting.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // U-Net with 9 input channels: 4 noisy latent + 4 masked source latent + 1 binary mask
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: INPUT_CHANNELS,
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
            inputChannels: INPUT_CHANNELS,
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

        return new PaintByExampleModel<T>(
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
            Name = "Paint-by-Example",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Paint-by-Example fills masked regions using exemplar images as visual references",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "latent-diffusion-inpainting");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("conditioning", "CLIP image encoder");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("input_channels", INPUT_CHANNELS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("exemplar_based", true);

        return metadata;
    }

    #endregion
}
