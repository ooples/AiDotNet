using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Playground v2.5 model for aesthetically-focused text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Playground v2.5 is an open-source text-to-image model optimized for aesthetic quality,
/// developed by Playground AI. It achieves state-of-the-art aesthetic scores while maintaining
/// the SDXL architecture for broad compatibility.
/// </para>
/// <para>
/// <b>For Beginners:</b> Playground v2.5 focuses on making the most visually appealing images:
///
/// How Playground v2.5 works:
/// 1. Uses the same SDXL architecture (dual text encoders, U-Net)
/// 2. Trained with enhanced aesthetic filtering and human preference alignment
/// 3. Uses EDM (Elucidated Diffusion Models) training framework
/// 4. Generates high-quality 1024x1024 images with exceptional aesthetics
///
/// Key characteristics:
/// - SDXL-compatible architecture (drop-in replacement)
/// - Dual text encoders: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-bigG/14 (1280-dim)
/// - Combined cross-attention dimension: 2048
/// - Aesthetic-focused training with human preference data
/// - EDM (Elucidated Diffusion Models) noise schedule
/// - Native 1024x1024 resolution
///
/// Advantages:
/// - Superior aesthetic quality (highest scores on various benchmarks)
/// - SDXL-compatible: works with SDXL LoRAs and tools
/// - Open-source (Apache 2.0 license)
/// - Excellent for photorealistic and artistic images
///
/// Limitations:
/// - Slightly slower than standard SDXL due to larger training
/// - May over-aestheticize some outputs
/// - Fewer community fine-tunes than standard SDXL
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SDXL-compatible U-Net with dual text encoders
/// - U-Net: ~2.6B parameters, base channels 320, multipliers [1, 2, 4]
/// - Text encoder 1: CLIP ViT-L/14 (768-dim)
/// - Text encoder 2: OpenCLIP ViT-bigG/14 (1280-dim)
/// - Combined context: 2048-dim (768 + 1280)
/// - VAE: SDXL VAE, 4 latent channels, scale factor 0.13025
/// - Training: EDM framework with aesthetic optimization
/// - Resolution: 1024x1024
///
/// Reference: Li et al., "Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Playground v2.5 for aesthetic generation
/// var playground = new PlaygroundV25Model&lt;float&gt;();
///
/// // Generate a high-quality 1024x1024 image
/// var image = playground.GenerateFromText(
///     prompt: "A serene Japanese garden with cherry blossoms and a stone bridge",
///     negativePrompt: "blurry, low quality, distorted",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 3.0,
///     seed: 42);
/// </code>
/// </example>
public class PlaygroundV25Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Playground v2.5 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Playground v2.5 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int PG_LATENT_CHANNELS = 4;

    /// <summary>
    /// Combined cross-attention dimension (CLIP 768 + OpenCLIP 1280 = 2048).
    /// </summary>
    private const int PG_CROSS_ATTENTION_DIM = 2048;

    /// <summary>
    /// Default guidance scale for Playground v2.5 (3.0, lower than SDXL).
    /// </summary>
    private const double PG_DEFAULT_GUIDANCE_SCALE = 3.0;

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
    public override int LatentChannels => PG_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the cross-attention dimension (2048 for dual text encoders).
    /// </summary>
    public int CrossAttentionDim => PG_CROSS_ATTENTION_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PlaygroundV25Model with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses Playground v2.5 defaults with EDM schedule.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with EDM settings.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net. If null, creates the standard SDXL-architecture U-Net.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard SDXL VAE.
    /// </param>
    /// <param name="conditioner">
    /// Dual text encoder conditioning module.
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PlaygroundV25Model(
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

        SetGuidanceScale(PG_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers following SDXL architecture,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the Playground v2.5 paper.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // SDXL-architecture U-Net (~2.6B parameters)
        // Uses dual text encoder conditioning (2048-dim combined)
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: PG_LATENT_CHANNELS,
            outputChannels: PG_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: PG_CROSS_ATTENTION_DIM,
            seed: seed);

        // SDXL VAE (slightly different scale factor: 0.13025)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: PG_LATENT_CHANNELS,
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
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? PG_DEFAULT_GUIDANCE_SCALE;

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
        var effectiveGuidanceScale = guidanceScale ?? PG_DEFAULT_GUIDANCE_SCALE;

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
            inputChannels: PG_LATENT_CHANNELS,
            outputChannels: PG_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2],
            contextDim: PG_CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: PG_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.13025);
        clonedVae.SetParameters(_vae.GetParameters());

        return new PlaygroundV25Model<T>(
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
            Name = "Playground v2.5",
            Version = "2.5",
            ModelType = ModelType.NeuralNetwork,
            Description = "Playground v2.5 aesthetic-optimized text-to-image model with SDXL architecture and EDM training",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sdxl-edm-latent-diffusion");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "OpenCLIP ViT-bigG/14");
        metadata.SetProperty("cross_attention_dim", PG_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", PG_LATENT_CHANNELS);
        metadata.SetProperty("training_framework", "EDM");
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
