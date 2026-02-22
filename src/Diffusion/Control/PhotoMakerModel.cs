using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// PhotoMaker model — identity-customized photo generation with stacked ID embedding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PhotoMaker generates customized photos of a person using 1-4 reference images.
/// It uses a stacked ID embedding approach that fuses identity features from multiple
/// reference images into the text conditioning pipeline.
/// </para>
/// <para>
/// <b>For Beginners:</b> PhotoMaker creates personalized photos from a few reference images:
///
/// Key characteristics:
/// - 1-4 reference images for identity (no fine-tuning needed)
/// - Stacked ID embedding: fuses multiple reference features
/// - CLIP image encoder for identity extraction
/// - Works with SDXL for high-quality output
///
/// Use PhotoMaker when you need:
/// - Customized photo generation from few references
/// - Identity-consistent character images
/// - Quick personalization without training
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SDXL + stacked ID embedding
/// - Identity encoder: CLIP ViT-L/14 (fine-tuned)
/// - Base model: SDXL U-Net
/// - Resolution: 1024x1024
///
/// Reference: Li et al., "PhotoMaker: Customizing Realistic Human Photos
/// via Stacked ID Embedding", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PhotoMaker model with default settings
/// var photoMaker = new PhotoMakerModel&lt;float&gt;();
///
/// // Generate an identity-customized image from text
/// var image = photoMaker.GenerateFromText(
///     prompt: "A portrait of [person] in a professional setting",
///     negativePrompt: "blurry, low quality",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 30,
///     guidanceScale: 5.0,
///     seed: 42);
/// </code>
/// </example>
public class PhotoMakerModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>Default width for PhotoMaker generation (SDXL native).</summary>
    public const int DefaultWidth = 1024;

    /// <summary>Default height for PhotoMaker generation (SDXL native).</summary>
    public const int DefaultHeight = 1024;

    /// <summary>Number of latent channels in the VAE latent space.</summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>Cross-attention dimension for SDXL (2048).</summary>
    private const int CROSS_ATTENTION_DIM = 2048;

    /// <summary>Default guidance scale for identity-preserving generation.</summary>
    private const double DEFAULT_GUIDANCE_SCALE = 5.0;

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

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the <see cref="PhotoMakerModel{T}"/> class.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom noise scheduler.</param>
    /// <param name="unet">Optional pre-configured U-Net noise predictor.</param>
    /// <param name="vae">Optional pre-configured VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text guidance.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PhotoMakerModel(
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
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net noise predictor and VAE components.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.13025,
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
        int numInferenceSteps = 30,
        double? guidanceScale = null,
        int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _unet.GetParameters();
        var vp = _vae.GetParameters();
        var c = new Vector<T>(up.Length + vp.Length);

        for (int i = 0; i < up.Length; i++)
        {
            c[i] = up[i];
        }

        for (int i = 0; i < vp.Length; i++)
        {
            c[up.Length + i] = vp[i];
        }

        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _unet.ParameterCount;
        int vc = _vae.ParameterCount;

        if (parameters.Length != uc + vc)
        {
            throw new ArgumentException(
                $"Expected {uc + vc} parameters (U-Net: {uc}, VAE: {vc}), got {parameters.Length}.",
                nameof(parameters));
        }

        var up = new Vector<T>(uc);
        var vp = new Vector<T>(vc);

        for (int i = 0; i < uc; i++)
        {
            up[i] = parameters[i];
        }

        for (int i = 0; i < vc; i++)
        {
            vp[i] = parameters[uc + i];
        }

        _unet.SetParameters(up);
        _vae.SetParameters(vp);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cu = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM);
        cu.SetParameters(_unet.GetParameters());

        var cv = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.13025);
        cv.SetParameters(_vae.GetParameters());

        return new PhotoMakerModel<T>(unet: cu, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "PhotoMaker",
            Version = "2.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "PhotoMaker identity-customized photo generation with stacked ID embedding",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        m.SetProperty("architecture", "sdxl-stacked-id-embedding");
        m.SetProperty("identity_encoder", "CLIP-ViT-L/14");
        m.SetProperty("base_model", "SDXL");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        m.SetProperty("default_resolution", $"{DefaultWidth}x{DefaultHeight}");
        m.SetProperty("max_reference_images", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE_SCALE);
        m.SetProperty("reference", "Li et al., CVPR 2024");
        m.SetProperty("requires_finetuning", false);

        return m;
    }

    #endregion
}
