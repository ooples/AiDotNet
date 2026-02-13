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
/// Kolors model — ChatGLM3-powered bilingual text-to-image model by Kwai.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kolors is Kwai's text-to-image model that uses ChatGLM3-6B as the text encoder,
/// providing strong bilingual (Chinese-English) text understanding. It builds on the
/// SDXL U-Net architecture with ChatGLM3's 4096-dim embeddings for cross-attention.
/// </para>
/// <para>
/// <b>For Beginners:</b> Kolors is like SDXL but with Chinese language understanding:
///
/// Key characteristics:
/// - ChatGLM3-6B text encoder: strong Chinese + English understanding
/// - SDXL-like U-Net backbone: proven architecture for quality
/// - 4096-dim cross-attention from ChatGLM3 embeddings
/// - Native 1024x1024 resolution
/// - Open-source with Apache 2.0 license
///
/// How Kolors works:
/// 1. Text goes through ChatGLM3-6B (6B parameter language model)
/// 2. 4096-dim embeddings provide rich text understanding
/// 3. SDXL-like U-Net denoises with cross-attention to embeddings
/// 4. VAE decodes to final image
///
/// Use Kolors when you need:
/// - Chinese text-to-image generation
/// - Open-source bilingual model
/// - SDXL-quality with multilingual support
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SDXL U-Net with ChatGLM3-6B text encoder
/// - Text encoder: ChatGLM3-6B (4096-dim, 65024 vocab)
/// - U-Net: ~2.6B parameters
/// - Resolution: 1024x1024
/// - Latent space: 4 channels, 8x downsampling
/// - Guidance scale: 5.0-7.5 recommended
///
/// Reference: Kwai, "Kolors: Effective Training of Diffusion Model for Photorealistic
/// Text-to-Image Synthesis", 2024
/// </para>
/// </remarks>
public class KolorsModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 4096;
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
    /// Initializes a new instance of KolorsModel with full customization support.
    /// </summary>
    public KolorsModel(
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
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
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
        // SDXL-like U-Net with ChatGLM3 cross-attention dimension
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
        int numInferenceSteps = 25,
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
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        clonedVae.SetParameters(_vae.GetParameters());

        return new KolorsModel<T>(
            unet: clonedUnet, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Kolors",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Kolors ChatGLM3-powered bilingual text-to-image model",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sdxl-unet-chatglm3");
        metadata.SetProperty("text_encoder", "ChatGLM3-6B");
        metadata.SetProperty("bilingual", true);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("license", "Apache-2.0");

        return metadata;
    }

    #endregion
}
