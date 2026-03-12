using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// SANA model for efficient high-resolution text-to-image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SANA uses a linear DiT architecture with efficient linear attention for high-resolution
/// generation (up to 4K). It achieves state-of-the-art quality with a compact 0.6B parameter
/// model through Deep Compression Autoencoder (DC-AE) with 32x spatial compression and
/// linear attention mechanisms that reduce compute from O(n^2) to O(n).
/// </para>
/// <para>
/// <b>For Beginners:</b> SANA generates high-quality images very efficiently. While other
/// models need billions of parameters, SANA achieves similar quality with only 600 million,
/// making it much faster and requiring less memory.
///
/// How SANA works:
/// 1. Text is encoded by a Gemma 2B language model (Google's text encoder)
/// 2. A 20-layer linear DiT processes text embeddings and noise with O(n) attention
/// 3. DC-AE with 32x spatial compression decodes tiny latents to full images
/// 4. Flow matching training enables efficient 20-step generation
///
/// Model variants:
/// - SANA-0.6B: Default, 600M parameter DiT
/// - SANA-1.6B: Larger 1.6B parameter variant for higher quality
///
/// Key characteristics:
/// - 0.6B parameters in the transformer (vs 2-12B for competitors)
/// - 32x spatial compression via DC-AE (vs 8x standard)
/// - Linear attention: O(n) instead of O(n^2) quadratic attention
/// - Gemma 2B text encoder for strong multilingual prompt understanding
/// - 32 latent channels for high information retention
/// - Up to 4096x4096 resolution generation
///
/// Advantages:
/// - Extremely efficient: 100x+ faster than Flux1/SDXL for similar quality
/// - Low VRAM: runs on consumer GPUs with 8GB VRAM
/// - High resolution: native 4K generation support
/// - Strong text rendering via Gemma encoder
///
/// Limitations:
/// - Less community ecosystem than Stable Diffusion family
/// - Fewer LoRA/ControlNet adaptations available
/// - 32x compression can lose very fine details at low resolution
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Linear DiT with efficient linear attention
/// - Transformer: 0.6B params, hidden 2240, 20 layers, 20 heads
/// - Text encoder: Gemma 2B (2048-dim embeddings)
/// - VAE: DC-AE with 32 latent channels, 32x spatial compression
/// - Training: Flow matching with linear noise schedule
/// - Default: 20 inference steps, guidance scale 4.5
/// - Resolution: 512x512 to 4096x4096 (aspect-ratio aware)
///
/// Reference: Xie et al., "SANA: Efficient High-Resolution Image Synthesis with
/// Linear Diffusion Transformers", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create default SANA 0.6B
/// var sana = new SANAModel&lt;float&gt;();
///
/// // Create larger 1.6B variant
/// var sanaLarge = new SANAModel&lt;float&gt;(variant: SANAVariant.Large);
///
/// // Generate an image
/// var image = sana.GenerateFromText(
///     prompt: "A serene mountain lake at sunset with reflections",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 20,
///     guidanceScale: 4.5,
///     seed: 42);
/// </code>
/// </example>
public class SANAModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SANA (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for SANA (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int SANA_LATENT_CHANNELS = 32;
    private const int SANA_HIDDEN_SIZE = 2240;
    private const int SANA_NUM_LAYERS = 20;
    private const int SANA_NUM_HEADS = 20;
    private const int SANA_CONTEXT_DIM = 2048;
    private const int SANA_DOWNSAMPLE_FACTOR = 32;
    private const double SANA_DEFAULT_GUIDANCE = 4.5;
    private const int SANA_DEFAULT_STEPS = 20;

    // Larger variant constants
    private const int SANA_LARGE_HIDDEN_SIZE = 3072;
    private const int SANA_LARGE_NUM_LAYERS = 32;

    #endregion

    #region Fields

    private EMMDiTPredictor<T> _predictor;
    private DeepCompressionVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly SANAVariant _variant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SANA_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the model variant.
    /// </summary>
    public SANAVariant Variant => _variant;

    /// <summary>
    /// Gets the spatial compression factor of the DC-AE (32x).
    /// </summary>
    public int CompressionFactor => SANA_DOWNSAMPLE_FACTOR;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of SANAModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses SANA flow matching defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, uses flow-matching DDIM.</param>
    /// <param name="predictor">Custom linear DiT noise predictor.</param>
    /// <param name="vae">Custom DC-AE (32x compression). If null, creates default DC-AE.</param>
    /// <param name="conditioner">Text encoder module. If null, no built-in conditioner.</param>
    /// <param name="variant">Model variant: Small (0.6B, default) or Large (1.6B).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SANAModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        EMMDiTPredictor<T>? predictor = null,
        DeepCompressionVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        SANAVariant variant = SANAVariant.Small,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 1.0,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        _variant = variant;

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(SANA_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        EMMDiTPredictor<T>? predictor,
        DeepCompressionVAE<T>? vae,
        int? seed)
    {
        bool isLarge = _variant == SANAVariant.Large;

        // Linear DiT with SANA-specific dimensions and Gemma context
        _predictor = predictor ?? new EMMDiTPredictor<T>(
            inputChannels: SANA_LATENT_CHANNELS,
            contextDim: SANA_CONTEXT_DIM,
            seed: seed);

        // DC-AE with 32x spatial compression and 32 latent channels
        _vae = vae ?? new DeepCompressionVAE<T>(
            inputChannels: 3,
            latentChannels: SANA_LATENT_CHANNELS,
            downsampleFactor: SANA_DOWNSAMPLE_FACTOR,
            baseChannels: 128,
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
        int numInferenceSteps = SANA_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SANA_DEFAULT_GUIDANCE;

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
        int numInferenceSteps = SANA_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? SANA_DEFAULT_GUIDANCE;

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
        var predictorParams = _predictor.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = predictorParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < predictorParams.Length; i++)
            combined[i] = predictorParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[predictorParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var predictorCount = _predictor.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != predictorCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {predictorCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var predictorParams = new Vector<T>(predictorCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < predictorCount; i++)
            predictorParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[predictorCount + i];

        _predictor.SetParameters(predictorParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedPredictor = new EMMDiTPredictor<T>(
            inputChannels: SANA_LATENT_CHANNELS,
            contextDim: SANA_CONTEXT_DIM);
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new DeepCompressionVAE<T>(
            inputChannels: 3,
            latentChannels: SANA_LATENT_CHANNELS,
            downsampleFactor: SANA_DOWNSAMPLE_FACTOR,
            baseChannels: 128);
        clonedVae.SetParameters(_vae.GetParameters());

        return new SANAModel<T>(
            predictor: clonedPredictor,
            vae: clonedVae,
            conditioner: _conditioner,
            variant: _variant);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        bool isLarge = _variant == SANAVariant.Large;
        var variantName = isLarge ? "1.6B" : "0.6B";
        var metadata = new ModelMetadata<T>
        {
            Name = $"SANA {variantName}",
            Version = variantName,
            ModelType = ModelType.NeuralNetwork,
            Description = $"SANA {variantName} linear DiT with DC-AE (32x compression) and Gemma text encoder for efficient high-resolution generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "linear-dit-dc-ae-flow-matching");
        metadata.SetProperty("base_model", "SANA");
        metadata.SetProperty("text_encoder", "Gemma 2B");
        metadata.SetProperty("context_dim", SANA_CONTEXT_DIM);
        metadata.SetProperty("hidden_size", isLarge ? SANA_LARGE_HIDDEN_SIZE : SANA_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", isLarge ? SANA_LARGE_NUM_LAYERS : SANA_NUM_LAYERS);
        metadata.SetProperty("num_heads", SANA_NUM_HEADS);
        metadata.SetProperty("latent_channels", SANA_LATENT_CHANNELS);
        metadata.SetProperty("compression_factor", SANA_DOWNSAMPLE_FACTOR);
        metadata.SetProperty("attention_type", "linear");
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("max_resolution", 4096);
        metadata.SetProperty("default_guidance_scale", SANA_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", SANA_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
