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
/// Stable Diffusion 2.0/2.1 model for text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Diffusion 2.x is the second generation of Stability AI's text-to-image model.
/// It uses OpenCLIP ViT-H/14 instead of the original CLIP ViT-L/14 text encoder,
/// and introduces v-prediction as the default prediction type.
/// </para>
/// <para>
/// <b>For Beginners:</b> SD 2.0/2.1 is an upgraded version of SD 1.5 with key differences:
///
/// How SD 2.x works:
/// 1. Your text prompt is encoded by OpenCLIP ViT-H/14 (1024-dim embeddings)
/// 2. These embeddings guide a U-Net (865M parameters) that denoises in latent space
/// 3. A VAE decodes the denoised latent into a 768x768 image (SD 2.0) or 512/768 (SD 2.1)
///
/// Key differences from SD 1.5:
/// - Text encoder: OpenCLIP ViT-H/14 (1024-dim) vs CLIP ViT-L/14 (768-dim)
/// - Prediction type: v-prediction vs epsilon-prediction
/// - Native resolution: 768x768 (SD 2.0) or 512x512/768x768 (SD 2.1)
/// - Removed NSFW content from training data
/// - Better at generating text in images
///
/// When to use SD 2.x:
/// - Need v-prediction for certain workflows
/// - Want OpenCLIP text encoder (different strengths than CLIP)
/// - Need 768x768 native resolution
///
/// Limitations:
/// - Smaller community ecosystem than SD 1.5
/// - Some users prefer SD 1.5's CLIP encoder for prompt adherence
/// - Not as widely supported by third-party tools
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: U-Net with cross-attention and time embedding
/// - Text encoder: OpenCLIP ViT-H/14 (632M parameters, 1024-dim, 77 max tokens)
/// - U-Net: 865M parameters, base channels 320, [1, 2, 4, 4] multipliers
/// - Cross-attention dimension: 1024 (matches OpenCLIP output)
/// - VAE: KL-regularized autoencoder, 4 latent channels, scale factor 0.18215
/// - Noise schedule: Scaled linear beta schedule, 1000 training timesteps
/// - Prediction type: v-prediction (velocity prediction)
///
/// Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var sd2 = new StableDiffusion2Model&lt;float&gt;();
///
/// // Generate a 768x768 image from text
/// var image = sd2.GenerateFromText(
///     prompt: "A photograph of a mountain landscape at golden hour",
///     negativePrompt: "blurry, low quality, distorted",
///     width: 768,
///     height: 768,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5,
///     seed: 42);
/// </code>
/// </example>
public class StableDiffusion2Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SD 2.x generation.
    /// </summary>
    public const int DefaultWidth = 768;

    /// <summary>
    /// Default image height for SD 2.x generation.
    /// </summary>
    public const int DefaultHeight = 768;

    private const int SD2_LATENT_CHANNELS = 4;
    private const int SD2_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Cross-attention dimension matching OpenCLIP ViT-H/14 output (1024).
    /// </summary>
    private const int SD2_CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Default guidance scale for SD 2.x (7.5).
    /// </summary>
    private const double SD2_DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _useVPrediction;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SD2_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the cross-attention dimension (1024 for SD 2.x, matching OpenCLIP ViT-H/14).
    /// </summary>
    public int CrossAttentionDim => SD2_CROSS_ATTENTION_DIM;

    /// <summary>
    /// Gets whether this model uses v-prediction (velocity prediction).
    /// </summary>
    public bool UsesVPrediction => _useVPrediction;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableDiffusion2Model with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses SD 2.x defaults: scaled linear beta [0.00085, 0.012], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with SD 2.x settings.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net. If null, creates the standard SD 2.x U-Net with 1024-dim cross-attention.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard SD 2.x VAE (same architecture as SD 1.5).
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically OpenCLIP ViT-H/14).
    /// </param>
    /// <param name="useVPrediction">
    /// Whether to use v-prediction (default: true for SD 2.x).
    /// SD 2.0 uses v-prediction, SD 2.1 supports both epsilon and v-prediction.
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StableDiffusion2Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        bool useVPrediction = true,
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
        _useVPrediction = useVPrediction;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(SD2_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers, using custom layers from the user
    /// if provided or creating industry-standard layers from the SD 2.x paper.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Use custom U-Net from user, or create industry-standard SD 2.x U-Net
        // Key difference: 1024-dim cross-attention (OpenCLIP ViT-H/14)
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: SD2_LATENT_CHANNELS,
            outputChannels: SD2_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: SD2_CROSS_ATTENTION_DIM,
            seed: seed);

        // Use custom VAE from user, or create industry-standard SD 2.x VAE
        // VAE architecture is the same as SD 1.5
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD2_LATENT_CHANNELS,
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
        var effectiveGuidanceScale = guidanceScale ?? SD2_DEFAULT_GUIDANCE_SCALE;

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
        var effectiveGuidanceScale = guidanceScale ?? SD2_DEFAULT_GUIDANCE_SCALE;

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
            inputChannels: SD2_LATENT_CHANNELS,
            outputChannels: SD2_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: SD2_CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD2_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new StableDiffusion2Model<T>(
            unet: clonedUnet,
            vae: clonedVae,
            conditioner: _conditioner,
            useVPrediction: _useVPrediction);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Stable Diffusion 2.1",
            Version = "2.1",
            ModelType = ModelType.NeuralNetwork,
            Description = "Stable Diffusion 2.x latent diffusion model with OpenCLIP ViT-H/14 text conditioning and v-prediction",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "latent-diffusion");
        metadata.SetProperty("text_encoder", "OpenCLIP ViT-H/14");
        metadata.SetProperty("cross_attention_dim", SD2_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", SD2_LATENT_CHANNELS);
        metadata.SetProperty("vae_scale_factor", SD2_VAE_SCALE_FACTOR);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("prediction_type", _useVPrediction ? "v_prediction" : "epsilon");

        return metadata;
    }

    #endregion
}
