using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// DALL-E 2 (unCLIP) model for text-to-image generation via CLIP-guided diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DALL-E 2 is a text-to-image model developed by OpenAI that uses a two-stage pipeline:
/// a prior that generates CLIP image embeddings from text, and a decoder that generates
/// images from those embeddings. This approach is also known as "unCLIP".
/// </para>
/// <para>
/// <b>For Beginners:</b> DALL-E 2 generates images through CLIP space:
///
/// How DALL-E 2 works:
/// 1. Text is encoded by CLIP ViT-L/14 text encoder (768-dim)
/// 2. A diffusion prior maps text embeddings → CLIP image embeddings
/// 3. A diffusion decoder generates 64x64 images from image embeddings
/// 4. Two upsampler stages scale to 256x256 → 1024x1024
///
/// Key characteristics:
/// - Two-stage: Diffusion prior + Diffusion decoder
/// - CLIP-guided: operates in CLIP embedding space
/// - Text encoder: CLIP ViT-L/14 (768-dim)
/// - Prior: Diffusion transformer mapping text→image embeddings
/// - Decoder: Modified GLIDE model, pixel-space diffusion
/// - Supports image variations (re-generate from CLIP embedding)
///
/// Advantages:
/// - Natural image variations through CLIP space manipulation
/// - Good compositional understanding
/// - Supports text-guided image editing
///
/// Limitations:
/// - Not open-source (proprietary to OpenAI)
/// - Superseded by DALL-E 3 in quality
/// - Sometimes struggles with text rendering
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Diffusion Prior + Diffusion Decoder (unCLIP)
/// - Prior: Transformer-based diffusion, maps CLIP text→image embeddings
/// - Decoder: Modified GLIDE U-Net, ~3.5B parameters, 64x64 base
/// - Text encoder: CLIP ViT-L/14 (768-dim embeddings)
/// - Image encoder: CLIP ViT-L/14 (768-dim, used for prior training)
/// - Noise schedule: Linear beta, 1000 training timesteps
/// - Upsampler: Two ADM upsampler stages (64→256→1024)
///
/// Reference: Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents", 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var dalle2 = new DallE2Model&lt;float&gt;();
///
/// // Generate a 64x64 base image
/// var image = dalle2.GenerateFromText(
///     prompt: "An astronaut riding a horse in the style of Andy Warhol",
///     negativePrompt: null,
///     width: 64,
///     height: 64,
///     numInferenceSteps: 64,
///     guidanceScale: 4.0,
///     seed: 42);
/// </code>
/// </example>
public class DallE2Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for DALL-E 2 base decoder (64x64).
    /// </summary>
    public const int DefaultWidth = 64;

    /// <summary>
    /// Default image height for DALL-E 2 base decoder (64x64).
    /// </summary>
    public const int DefaultHeight = 64;

    private const int DALLE2_PIXEL_CHANNELS = 3;

    /// <summary>
    /// CLIP embedding dimension (768 for ViT-L/14).
    /// </summary>
    private const int DALLE2_CLIP_DIM = 768;

    /// <summary>
    /// Default guidance scale for DALL-E 2 (4.0).
    /// </summary>
    private const double DALLE2_DEFAULT_GUIDANCE_SCALE = 4.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _priorUnet;
    private UNetNoisePredictor<T> _decoderUnet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _decoderUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => DALLE2_PIXEL_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _priorUnet.ParameterCount + _decoderUnet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the diffusion prior that maps text embeddings to CLIP image embeddings.
    /// </summary>
    public INoisePredictor<T> DiffusionPrior => _priorUnet;

    /// <summary>
    /// Gets the CLIP embedding dimension (768).
    /// </summary>
    public int ClipDimension => DALLE2_CLIP_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of DallE2Model with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses DALL-E 2 defaults: linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with DALL-E 2 settings.
    /// </param>
    /// <param name="priorUnet">
    /// Custom diffusion prior. If null, creates the standard CLIP embedding diffusion prior.
    /// </param>
    /// <param name="decoderUnet">
    /// Custom decoder U-Net (modified GLIDE). If null, creates the standard ~3.5B parameter decoder.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates a minimal pixel-space VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically CLIP ViT-L/14).
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DallE2Model(
        NeuralNetworkArchitecture<T>? architecture = null,
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
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(priorUnet, decoderUnet, vae, seed);

        SetGuidanceScale(DALLE2_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the diffusion prior, decoder U-Net, and optional VAE,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the unCLIP/DALL-E 2 paper.
    /// </summary>
    [MemberNotNull(nameof(_priorUnet), nameof(_decoderUnet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? priorUnet,
        UNetNoisePredictor<T>? decoderUnet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Diffusion Prior: Maps CLIP text embeddings (768) to CLIP image embeddings (768)
        // Uses a transformer-based architecture
        _priorUnet = priorUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: DALLE2_CLIP_DIM,
            outputChannels: DALLE2_CLIP_DIM,
            baseChannels: 256,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [2, 1],
            contextDim: DALLE2_CLIP_DIM,
            seed: seed);

        // Decoder: Modified GLIDE U-Net (~3.5B parameters)
        // Generates 64x64 RGB images conditioned on CLIP image embeddings
        _decoderUnet = decoderUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: DALLE2_PIXEL_CHANNELS,
            outputChannels: DALLE2_PIXEL_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: DALLE2_CLIP_DIM,
            seed: seed);

        // Minimal pixel-space VAE
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: DALLE2_PIXEL_CHANNELS,
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
        int numInferenceSteps = 64,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? DALLE2_DEFAULT_GUIDANCE_SCALE;

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
        int numInferenceSteps = 64,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? DALLE2_DEFAULT_GUIDANCE_SCALE;

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
            inputChannels: DALLE2_CLIP_DIM,
            outputChannels: DALLE2_CLIP_DIM,
            baseChannels: 256,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [2, 1],
            contextDim: DALLE2_CLIP_DIM);
        clonedPrior.SetParameters(_priorUnet.GetParameters());

        var clonedDecoder = new UNetNoisePredictor<T>(
            inputChannels: DALLE2_PIXEL_CHANNELS,
            outputChannels: DALLE2_PIXEL_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: DALLE2_CLIP_DIM);
        clonedDecoder.SetParameters(_decoderUnet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: DALLE2_PIXEL_CHANNELS,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 1.0);
        clonedVae.SetParameters(_vae.GetParameters());

        return new DallE2Model<T>(
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
            Name = "DALL-E 2",
            Version = "2.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "DALL-E 2 (unCLIP) two-stage text-to-image model with diffusion prior and GLIDE decoder",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unclip-diffusion");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("clip_dim", DALLE2_CLIP_DIM);
        metadata.SetProperty("base_resolution", 64);
        metadata.SetProperty("final_resolution", 1024);
        metadata.SetProperty("pixel_channels", DALLE2_PIXEL_CHANNELS);

        return metadata;
    }

    #endregion
}
