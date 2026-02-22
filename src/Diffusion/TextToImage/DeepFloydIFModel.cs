using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// DeepFloyd IF model for cascaded text-to-image generation in pixel space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeepFloyd IF is a three-stage cascaded diffusion model that operates in pixel space
/// (not latent space), developed by DeepFloyd (Stability AI). It uses a frozen T5-XXL
/// text encoder for exceptional text understanding and prompt adherence.
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepFloyd IF generates images by progressively upscaling:
///
/// How DeepFloyd IF works:
/// 1. Stage I: Generates a 64x64 pixel image from text (using T5-XXL embeddings)
/// 2. Stage II: Upscales 64x64 → 256x256 with text-guided super-resolution
/// 3. Stage III: Upscales 256x256 → 1024x1024 (optional, non-diffusion upscaler)
///
/// Key characteristics:
/// - Pixel-space diffusion (no VAE/latent space for Stages I and II)
/// - Text encoder: Frozen T5-XXL (4.7B parameters, 4096-dim embeddings)
/// - Stage I: ~900M parameters, 64x64 output
/// - Stage II: ~450M parameters, 256x256 output
/// - Stage III: Non-diffusion upscaler to 1024x1024
/// - Exceptional text rendering in images
///
/// Advantages:
/// - Best-in-class text rendering (can write legible text)
/// - Exceptional prompt adherence from T5-XXL
/// - Pixel-space avoids VAE artifacts
///
/// Limitations:
/// - Very large memory requirement (T5-XXL alone is ~10GB)
/// - Slower than latent diffusion models
/// - Multi-stage pipeline adds complexity
/// - Restricted license (not fully open-source)
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Cascaded pixel-space diffusion
/// - Stage I: U-Net, ~900M parameters, 64x64 output, 3 RGB channels
/// - Stage II: U-Net, ~450M parameters, 256x256 output, 6 channels (3 RGB + 3 low-res input)
/// - Text encoder: Frozen T5-XXL (4.7B params, 4096-dim, 256 max tokens)
/// - Noise schedule: Cosine beta schedule, 1000 training timesteps
/// - Prediction type: Epsilon prediction with dynamic thresholding
///
/// Reference: Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", 2022
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var deepfloyd = new DeepFloydIFModel&lt;float&gt;();
///
/// // Generate a 64x64 base image (Stage I)
/// var image = deepfloyd.GenerateFromText(
///     prompt: "A sign that says 'Hello World' in neon lights",
///     negativePrompt: "blurry, low quality",
///     width: 64,
///     height: 64,
///     numInferenceSteps: 100,
///     guidanceScale: 7.0,
///     seed: 42);
/// </code>
/// </example>
public class DeepFloydIFModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Stage I (64x64 base generation).
    /// </summary>
    public const int DefaultWidth = 64;

    /// <summary>
    /// Default image height for Stage I (64x64 base generation).
    /// </summary>
    public const int DefaultHeight = 64;

    private const int IF_PIXEL_CHANNELS = 3;
    private const int IF_STAGE2_INPUT_CHANNELS = 6;

    /// <summary>
    /// Cross-attention dimension matching T5-XXL output (4096).
    /// </summary>
    private const int IF_CROSS_ATTENTION_DIM = 4096;

    /// <summary>
    /// Default guidance scale for DeepFloyd IF (7.0).
    /// </summary>
    private const double IF_DEFAULT_GUIDANCE_SCALE = 7.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _stageIUnet;
    private UNetNoisePredictor<T> _stageIIUnet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _useDynamicThresholding;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _stageIUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => IF_PIXEL_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _stageIUnet.ParameterCount + _stageIIUnet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the Stage II super-resolution noise predictor.
    /// </summary>
    public INoisePredictor<T> StageIINoisePredictor => _stageIIUnet;

    /// <summary>
    /// Gets the cross-attention dimension (4096 for T5-XXL).
    /// </summary>
    public int CrossAttentionDim => IF_CROSS_ATTENTION_DIM;

    /// <summary>
    /// Gets whether dynamic thresholding is enabled.
    /// </summary>
    public bool UsesDynamicThresholding => _useDynamicThresholding;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of DeepFloydIFModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses DeepFloyd IF defaults: cosine beta schedule, 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler with IF settings.
    /// </param>
    /// <param name="stageIUnet">
    /// Custom Stage I U-Net (64x64 generation). If null, creates the standard ~900M parameter model.
    /// </param>
    /// <param name="stageIIUnet">
    /// Custom Stage II U-Net (256x256 super-resolution). If null, creates the standard ~450M parameter model.
    /// </param>
    /// <param name="vae">
    /// Custom VAE for optional latent encoding. If null, creates a pass-through pixel-space VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module (typically T5-XXL).
    /// </param>
    /// <param name="useDynamicThresholding">
    /// Whether to use dynamic thresholding during denoising (default: true).
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DeepFloydIFModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? stageIUnet = null,
        UNetNoisePredictor<T>? stageIIUnet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        bool useDynamicThresholding = true,
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
        _useDynamicThresholding = useDynamicThresholding;

        InitializeLayers(stageIUnet, stageIIUnet, vae, seed);

        SetGuidanceScale(IF_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the Stage I (64x64), Stage II (256x256) U-Nets, and optional VAE,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the Imagen/DeepFloyd IF paper.
    /// </summary>
    [MemberNotNull(nameof(_stageIUnet), nameof(_stageIIUnet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? stageIUnet,
        UNetNoisePredictor<T>? stageIIUnet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Stage I: Base generation model (~900M parameters)
        // Generates 64x64 RGB images directly in pixel space
        _stageIUnet = stageIUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: IF_PIXEL_CHANNELS,
            outputChannels: IF_PIXEL_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: IF_CROSS_ATTENTION_DIM,
            seed: seed);

        // Stage II: Super-resolution model (~450M parameters)
        // Upscales 64x64 → 256x256, takes concatenated [noisy_256, low_res_64] input
        _stageIIUnet = stageIIUnet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: IF_STAGE2_INPUT_CHANNELS,
            outputChannels: IF_PIXEL_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: IF_CROSS_ATTENTION_DIM,
            seed: seed);

        // Pixel-space VAE (identity-like, 1:1 scale factor for pixel-space models)
        // DeepFloyd IF operates in pixel space, so the VAE is minimal
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IF_PIXEL_CHANNELS,
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
        var effectiveGuidanceScale = guidanceScale ?? IF_DEFAULT_GUIDANCE_SCALE;

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
        var effectiveGuidanceScale = guidanceScale ?? IF_DEFAULT_GUIDANCE_SCALE;

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
        var stage1Params = _stageIUnet.GetParameters();
        var stage2Params = _stageIIUnet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = stage1Params.Length + stage2Params.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        var offset = 0;
        for (int i = 0; i < stage1Params.Length; i++)
        {
            combined[offset + i] = stage1Params[i];
        }
        offset += stage1Params.Length;

        for (int i = 0; i < stage2Params.Length; i++)
        {
            combined[offset + i] = stage2Params[i];
        }
        offset += stage2Params.Length;

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var stage1Count = _stageIUnet.ParameterCount;
        var stage2Count = _stageIIUnet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != stage1Count + stage2Count + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {stage1Count + stage2Count + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var stage1Params = new Vector<T>(stage1Count);
        var stage2Params = new Vector<T>(stage2Count);
        var vaeParams = new Vector<T>(vaeCount);

        var offset = 0;
        for (int i = 0; i < stage1Count; i++)
        {
            stage1Params[i] = parameters[offset + i];
        }
        offset += stage1Count;

        for (int i = 0; i < stage2Count; i++)
        {
            stage2Params[i] = parameters[offset + i];
        }
        offset += stage2Count;

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset + i];
        }

        _stageIUnet.SetParameters(stage1Params);
        _stageIIUnet.SetParameters(stage2Params);
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
        var clonedStage1 = new UNetNoisePredictor<T>(
            inputChannels: IF_PIXEL_CHANNELS,
            outputChannels: IF_PIXEL_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: IF_CROSS_ATTENTION_DIM);
        clonedStage1.SetParameters(_stageIUnet.GetParameters());

        var clonedStage2 = new UNetNoisePredictor<T>(
            inputChannels: IF_STAGE2_INPUT_CHANNELS,
            outputChannels: IF_PIXEL_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: IF_CROSS_ATTENTION_DIM);
        clonedStage2.SetParameters(_stageIIUnet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IF_PIXEL_CHANNELS,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 1.0);
        clonedVae.SetParameters(_vae.GetParameters());

        return new DeepFloydIFModel<T>(
            stageIUnet: clonedStage1,
            stageIIUnet: clonedStage2,
            vae: clonedVae,
            conditioner: _conditioner,
            useDynamicThresholding: _useDynamicThresholding);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DeepFloyd IF",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "DeepFloyd IF cascaded pixel-space diffusion model with T5-XXL text encoder",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-pixel-diffusion");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("cross_attention_dim", IF_CROSS_ATTENTION_DIM);
        metadata.SetProperty("stage1_resolution", 64);
        metadata.SetProperty("stage2_resolution", 256);
        metadata.SetProperty("stage3_resolution", 1024);
        metadata.SetProperty("pixel_channels", IF_PIXEL_CHANNELS);
        metadata.SetProperty("dynamic_thresholding", _useDynamicThresholding);

        return metadata;
    }

    #endregion
}
