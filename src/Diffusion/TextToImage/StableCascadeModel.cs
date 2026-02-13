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
/// Stable Cascade (Würstchen v3) model for high-resolution text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Cascade is a three-stage cascaded latent diffusion model developed by Stability AI,
/// based on the Würstchen architecture. It achieves extreme compression (42:1 spatial ratio)
/// allowing fast, high-quality 1024x1024 generation with lower compute requirements.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stable Cascade generates images in three stages, like a relay race:
///
/// How Stable Cascade works:
/// 1. Stage C (Prior): Generates a tiny 24x24 latent from your text prompt
/// 2. Stage B (Decoder): Expands the 24x24 latent to a 256x256 latent
/// 3. Stage A (VQGAN): Decodes the 256x256 latent to a full 1024x1024 image
///
/// Key characteristics:
/// - Three-stage cascade: Stage C (prior) → Stage B (decoder) → Stage A (VQGAN)
/// - Extreme compression: 42:1 spatial ratio (vs 8:1 for SD 1.5)
/// - Stage C operates in a very small 24×24 latent space (24 channels)
/// - Stage B: ~700M parameters, denoising diffusion
/// - Stage A: VQGAN decoder (frozen, non-diffusion)
/// - Text encoder: CLIP ViT-G/14 (1280-dim embeddings)
/// - Native resolution: 1024x1024
///
/// Advantages over Stable Diffusion:
/// - Much faster training and inference due to extreme compression
/// - Lower VRAM requirements for training
/// - Native 1024x1024 without quality degradation
/// - Better text-image alignment
///
/// Limitations:
/// - Smaller community ecosystem than SD 1.5/SDXL
/// - Three-stage pipeline is more complex to customize
/// - Fewer fine-tuned models available
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Three-stage cascaded latent diffusion
/// - Stage C: Würstchen prior, 1B parameters, 24-channel 24×24 latent
/// - Stage B: Würstchen decoder, ~700M parameters, 4-channel 256×256 latent
/// - Stage A: VQGAN (EfficientNet-based), frozen during training
/// - Text encoder: CLIP ViT-G/14 (1280-dim, 77 max tokens)
/// - Compression ratio: 42:1 spatial (1024→24)
/// - Noise schedule: Linear beta, 1000 training timesteps
/// - Prediction type: Epsilon prediction
///
/// Reference: Pernias et al., "Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var cascade = new StableCascadeModel&lt;float&gt;();
///
/// // Generate a 1024x1024 image from text
/// var image = cascade.GenerateFromText(
///     prompt: "A majestic castle on a cliff overlooking the sea",
///     negativePrompt: "blurry, low quality",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 30,
///     guidanceScale: 4.0,
///     seed: 42);
/// </code>
/// </example>
public class StableCascadeModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Stable Cascade generation.
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Stable Cascade generation.
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int CASCADE_LATENT_CHANNELS = 24;
    private const int CASCADE_STAGE_B_LATENT_CHANNELS = 4;
    private const int CASCADE_VAE_SCALE_FACTOR = 42;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-G/14 output (1280).
    /// </summary>
    private const int CASCADE_CROSS_ATTENTION_DIM = 1280;

    /// <summary>
    /// Default guidance scale for Stable Cascade (4.0, lower than SD due to better alignment).
    /// </summary>
    private const double CASCADE_DEFAULT_GUIDANCE_SCALE = 4.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _priorUnet;
    private UNetNoisePredictor<T> _decoderUnet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _priorUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => CASCADE_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _priorUnet.ParameterCount + _decoderUnet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the Stage B decoder noise predictor.
    /// </summary>
    public INoisePredictor<T> DecoderNoisePredictor => _decoderUnet;

    /// <summary>
    /// Gets the cross-attention dimension (1280 for CLIP ViT-G/14).
    /// </summary>
    public int CrossAttentionDim => CASCADE_CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableCascadeModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses Stable Cascade defaults: linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with Stable Cascade settings.
    /// </param>
    /// <param name="priorUnet">
    /// Custom Stage C prior U-Net. If null, creates the standard ~1B parameter prior.
    /// </param>
    /// <param name="decoderUnet">
    /// Custom Stage B decoder U-Net. If null, creates the standard ~700M parameter decoder.
    /// </param>
    /// <param name="vae">
    /// Custom Stage A VQGAN/VAE. If null, creates the standard Stable Cascade VQGAN.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically CLIP ViT-G/14).
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StableCascadeModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? priorUnet = null,
        UNetNoisePredictor<T>? decoderUnet = null,
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
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;

        InitializeLayers(priorUnet, decoderUnet, vae, seed);

        SetGuidanceScale(CASCADE_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Stage C prior, Stage B decoder, and Stage A VQGAN layers,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the Würstchen paper.
    /// </summary>
    [MemberNotNull(nameof(_priorUnet), nameof(_decoderUnet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? priorUnet,
        UNetNoisePredictor<T>? decoderUnet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Stage C: Prior U-Net (~1B parameters)
        // Generates 24-channel 24×24 latent from text conditioning
        _priorUnet = priorUnet ?? new UNetNoisePredictor<T>(
            inputChannels: CASCADE_LATENT_CHANNELS,
            outputChannels: CASCADE_LATENT_CHANNELS,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: CASCADE_CROSS_ATTENTION_DIM,
            seed: seed);

        // Stage B: Decoder U-Net (~700M parameters)
        // Expands prior latent to 4-channel 256×256 latent
        _decoderUnet = decoderUnet ?? new UNetNoisePredictor<T>(
            inputChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            outputChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CASCADE_CROSS_ATTENTION_DIM,
            seed: seed);

        // Stage A: VQGAN decoder (frozen during training)
        // Decodes 4-channel 256×256 latent to 1024×1024 image
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.3611,
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
        var effectiveGuidanceScale = guidanceScale ?? CASCADE_DEFAULT_GUIDANCE_SCALE;

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
        int numInferenceSteps = 30,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? CASCADE_DEFAULT_GUIDANCE_SCALE;

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
        var priorParams = _priorUnet.GetParameters();
        var decoderParams = _decoderUnet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = priorParams.Length + decoderParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        var offset = 0;
        for (int i = 0; i < priorParams.Length; i++)
        {
            combined[offset + i] = priorParams[i];
        }
        offset += priorParams.Length;

        for (int i = 0; i < decoderParams.Length; i++)
        {
            combined[offset + i] = decoderParams[i];
        }
        offset += decoderParams.Length;

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var priorCount = _priorUnet.ParameterCount;
        var decoderCount = _decoderUnet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != priorCount + decoderCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {priorCount + decoderCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var priorParams = new Vector<T>(priorCount);
        var decoderParams = new Vector<T>(decoderCount);
        var vaeParams = new Vector<T>(vaeCount);

        var offset = 0;
        for (int i = 0; i < priorCount; i++)
        {
            priorParams[i] = parameters[offset + i];
        }
        offset += priorCount;

        for (int i = 0; i < decoderCount; i++)
        {
            decoderParams[i] = parameters[offset + i];
        }
        offset += decoderCount;

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset + i];
        }

        _priorUnet.SetParameters(priorParams);
        _decoderUnet.SetParameters(decoderParams);
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
        var clonedPrior = new UNetNoisePredictor<T>(
            inputChannels: CASCADE_LATENT_CHANNELS,
            outputChannels: CASCADE_LATENT_CHANNELS,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: CASCADE_CROSS_ATTENTION_DIM);
        clonedPrior.SetParameters(_priorUnet.GetParameters());

        var clonedDecoder = new UNetNoisePredictor<T>(
            inputChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            outputChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: CASCADE_CROSS_ATTENTION_DIM);
        clonedDecoder.SetParameters(_decoderUnet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: CASCADE_STAGE_B_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.3611);
        clonedVae.SetParameters(_vae.GetParameters());

        return new StableCascadeModel<T>(
            priorUnet: clonedPrior,
            decoderUnet: clonedDecoder,
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
            Name = "Stable Cascade",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Stable Cascade (Würstchen v3) three-stage cascaded latent diffusion model with extreme 42:1 compression",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-latent-diffusion");
        metadata.SetProperty("text_encoder", "CLIP ViT-G/14");
        metadata.SetProperty("cross_attention_dim", CASCADE_CROSS_ATTENTION_DIM);
        metadata.SetProperty("prior_latent_channels", CASCADE_LATENT_CHANNELS);
        metadata.SetProperty("decoder_latent_channels", CASCADE_STAGE_B_LATENT_CHANNELS);
        metadata.SetProperty("compression_ratio", CASCADE_VAE_SCALE_FACTOR);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
