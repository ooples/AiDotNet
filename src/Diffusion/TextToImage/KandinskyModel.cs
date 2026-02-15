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
/// Kandinsky 2.2/3.0 model for text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kandinsky is a two-stage text-to-image model developed by Sber AI and AI Forever.
/// It uses a prior model (diffusion or transformer) to map text embeddings to image embeddings,
/// then a latent diffusion decoder to generate images from those embeddings.
/// </para>
/// <para>
/// <b>For Beginners:</b> Kandinsky works in two phases, like a translator:
///
/// How Kandinsky works:
/// 1. Prior stage: Translates text embeddings (CLIP) into image embeddings
/// 2. Decoder stage: A latent diffusion model generates images conditioned on the image embeddings
///
/// Key characteristics:
/// - Two-stage pipeline: Prior → Decoder
/// - Prior: Diffusion-based mapping from text to image embedding space
/// - Decoder: U-Net latent diffusion model (similar to SD architecture)
/// - Text encoder: CLIP ViT-G/14 (1280-dim) + multilingual XLM-RoBERTa
/// - Image encoder: CLIP ViT-G/14 (used to train the prior)
/// - Native resolution: 1024x1024 (Kandinsky 3.0) or 512x512 (Kandinsky 2.2)
/// - VAE: Movq (MoVQ-GAN), 4 latent channels
///
/// Advantages:
/// - Strong multilingual support through XLM-RoBERTa
/// - Two-stage architecture allows separate optimization of text→embedding and embedding→image
/// - Good prompt adherence through CLIP-space prior
///
/// Limitations:
/// - Two-stage pipeline adds latency
/// - Smaller ecosystem than Stable Diffusion
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Prior + Latent Diffusion Decoder
/// - Prior: Diffusion transformer, maps CLIP text→image embeddings
/// - Decoder U-Net: ~1.2B parameters, 4-channel latent, channel multipliers [1, 2, 4, 4]
/// - Text encoder: CLIP ViT-G/14 (1280-dim)
/// - VAE: MoVQ-GAN, 4 latent channels, scale factor 0.18215
/// - Noise schedule: Linear beta schedule, 1000 training timesteps
///
/// Reference: Razzhigaev et al., "Kandinsky: an Improved Text-to-Image Synthesis with Image Prior and Latent Diffusion", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var kandinsky = new KandinskyModel&lt;float&gt;();
///
/// // Generate a 1024x1024 image from text
/// var image = kandinsky.GenerateFromText(
///     prompt: "An oil painting of a Russian winter landscape",
///     negativePrompt: "blurry, deformed",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 4.0,
///     seed: 42);
/// </code>
/// </example>
public class KandinskyModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Kandinsky 3.0 generation.
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Kandinsky 3.0 generation.
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int KANDINSKY_LATENT_CHANNELS = 4;
    private const int KANDINSKY_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Cross-attention dimension matching CLIP ViT-G/14 output (1280).
    /// </summary>
    private const int KANDINSKY_CROSS_ATTENTION_DIM = 1280;

    /// <summary>
    /// Dimension of the CLIP image embedding space for the prior.
    /// </summary>
    private const int KANDINSKY_IMAGE_EMBEDDING_DIM = 1280;

    /// <summary>
    /// Default guidance scale for Kandinsky (4.0).
    /// </summary>
    private const double KANDINSKY_DEFAULT_GUIDANCE_SCALE = 4.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _priorUnet;
    private UNetNoisePredictor<T> _decoderUnet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly string _version;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _decoderUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => KANDINSKY_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _priorUnet.ParameterCount + _decoderUnet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the prior model that maps text embeddings to image embeddings.
    /// </summary>
    public INoisePredictor<T> PriorModel => _priorUnet;

    /// <summary>
    /// Gets the model version ("2.2" or "3.0").
    /// </summary>
    public string Version => _version;

    /// <summary>
    /// Gets the cross-attention dimension (1280 for CLIP ViT-G/14).
    /// </summary>
    public int CrossAttentionDim => KANDINSKY_CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of KandinskyModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses Kandinsky defaults: linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with Kandinsky settings.
    /// </param>
    /// <param name="priorUnet">
    /// Custom prior U-Net that maps text to image embeddings. If null, creates the standard prior.
    /// </param>
    /// <param name="decoderUnet">
    /// Custom decoder U-Net for latent diffusion. If null, creates the standard ~1.2B parameter decoder.
    /// </param>
    /// <param name="vae">
    /// Custom VAE (MoVQ-GAN). If null, creates the standard Kandinsky VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically CLIP ViT-G/14).
    /// </param>
    /// <param name="version">Model version: "2.2" or "3.0" (default: "3.0").</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public KandinskyModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? priorUnet = null,
        UNetNoisePredictor<T>? decoderUnet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        string version = "3.0",
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
        _version = version;

        InitializeLayers(priorUnet, decoderUnet, vae, seed);

        SetGuidanceScale(KANDINSKY_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the prior, decoder U-Net, and MoVQ-GAN VAE layers,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the Kandinsky paper.
    /// </summary>
    [MemberNotNull(nameof(_priorUnet), nameof(_decoderUnet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? priorUnet,
        UNetNoisePredictor<T>? decoderUnet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Prior: Diffusion model in CLIP image embedding space
        // Maps text embeddings (1280-dim) to image embeddings (1280-dim)
        _priorUnet = priorUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: KANDINSKY_IMAGE_EMBEDDING_DIM,
            outputChannels: KANDINSKY_IMAGE_EMBEDDING_DIM,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [2, 1],
            contextDim: KANDINSKY_CROSS_ATTENTION_DIM,
            seed: seed);

        // Decoder: Latent diffusion U-Net (~1.2B parameters)
        // Generates images conditioned on image embeddings from the prior
        _decoderUnet = decoderUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: KANDINSKY_LATENT_CHANNELS,
            outputChannels: KANDINSKY_LATENT_CHANNELS,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: KANDINSKY_CROSS_ATTENTION_DIM,
            seed: seed);

        // MoVQ-GAN: Vector-quantized GAN decoder
        // Same latent structure as SD VAE but uses vector quantization
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: KANDINSKY_LATENT_CHANNELS,
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
        var effectiveGuidanceScale = guidanceScale ?? KANDINSKY_DEFAULT_GUIDANCE_SCALE;

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
        var effectiveGuidanceScale = guidanceScale ?? KANDINSKY_DEFAULT_GUIDANCE_SCALE;

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
            inputChannels: KANDINSKY_IMAGE_EMBEDDING_DIM,
            outputChannels: KANDINSKY_IMAGE_EMBEDDING_DIM,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [2, 1],
            contextDim: KANDINSKY_CROSS_ATTENTION_DIM);
        clonedPrior.SetParameters(_priorUnet.GetParameters());

        var clonedDecoder = new UNetNoisePredictor<T>(
            inputChannels: KANDINSKY_LATENT_CHANNELS,
            outputChannels: KANDINSKY_LATENT_CHANNELS,
            baseChannels: 384,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: KANDINSKY_CROSS_ATTENTION_DIM);
        clonedDecoder.SetParameters(_decoderUnet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: KANDINSKY_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new KandinskyModel<T>(
            priorUnet: clonedPrior,
            decoderUnet: clonedDecoder,
            vae: clonedVae,
            conditioner: _conditioner,
            version: _version);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"Kandinsky {_version}",
            Version = _version,
            ModelType = ModelType.NeuralNetwork,
            Description = $"Kandinsky {_version} two-stage text-to-image model with CLIP prior and latent diffusion decoder",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "prior-latent-diffusion");
        metadata.SetProperty("text_encoder", "CLIP ViT-G/14");
        metadata.SetProperty("cross_attention_dim", KANDINSKY_CROSS_ATTENTION_DIM);
        metadata.SetProperty("image_embedding_dim", KANDINSKY_IMAGE_EMBEDDING_DIM);
        metadata.SetProperty("latent_channels", KANDINSKY_LATENT_CHANNELS);
        metadata.SetProperty("vae_type", "MoVQ-GAN");
        metadata.SetProperty("vae_scale_factor", KANDINSKY_VAE_SCALE_FACTOR);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
