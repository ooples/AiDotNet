using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// PixArt-α model for efficient high-quality text-to-image generation using DiT architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PixArt-α is an efficient text-to-image diffusion model that uses a Diffusion Transformer (DiT)
/// architecture. It achieves comparable quality to larger models like Stable Diffusion XL while
/// being significantly faster and more resource-efficient.
/// </para>
/// <para>
/// <b>For Beginners:</b> PixArt-α is like a sports car version of image generation:
///
/// Key advantages over traditional models:
/// - 10x faster training than Stable Diffusion
/// - Much more parameter-efficient
/// - Uses transformer blocks instead of U-Net
/// - T5-XXL text encoder for better prompt understanding
///
/// How PixArt-α works:
/// 1. Your prompt goes through a T5-XXL text encoder (larger = better understanding)
/// 2. The DiT (Diffusion Transformer) denoises using attention blocks
/// 3. Each block uses cross-attention to the text embedding
/// 4. The output is decoded by a VAE into an image
///
/// Example use cases:
/// - Fast prototyping (quick iterations)
/// - Resource-constrained environments (smaller models)
/// - High-quality generation without massive GPU requirements
/// - Applications requiring many generations
///
/// When to choose PixArt-α:
/// - You need faster generation than SDXL
/// - You want good quality without 70B+ model overhead
/// - Your prompts are complex (T5 encoder helps)
/// - You're doing many generations in batch
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Diffusion Transformer (DiT) with AdaLN-single
/// - Text encoder: T5-XXL (4.3B parameters, optional smaller variants)
/// - Native resolutions: 256x256 to 1024x1024
/// - Latent space: 4 channels, 8x spatial downsampling
/// - Training: Decomposed training strategy for efficiency
///
/// Architecture innovations:
/// - Cross-attention in every DiT block
/// - AdaLN-single for timestep conditioning (not AdaLN-Zero)
/// - Efficient attention patterns
/// - Multi-resolution training support
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PixArt-α model
/// var pixart = new PixArtModel&lt;float&gt;();
///
/// // Generate an image efficiently
/// var image = pixart.GenerateFromText(
///     prompt: "A serene Japanese garden with cherry blossoms",
///     negativePrompt: "blurry, low quality",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 20,
///     guidanceScale: 4.5,
///     seed: 42);
///
/// // Use different resolutions
/// var portrait = pixart.GenerateFromText(
///     prompt: "Portrait of an astronaut",
///     width: 768,
///     height: 1024);
///
/// // Generate multiple images with different seeds
/// var variations = pixart.GenerateVariations(
///     prompt: "Abstract art with vibrant colors",
///     count: 4);
/// </code>
/// </example>
public class PixArtModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default model size variant.
    /// </summary>
    /// <remarks>
    /// The "alpha" variant is the standard PixArt model with 600M parameters and 1024px default resolution.
    /// Other supported variants include "sigma" (512px), "delta" (256px), and "xl" (larger 1024px).
    /// </remarks>
    public const string DefaultModelSize = "alpha";

    /// <summary>
    /// Standard PixArt latent channels.
    /// </summary>
    /// <remarks>
    /// PixArt uses 4 latent channels matching the standard VAE latent space dimensionality.
    /// This is consistent with other latent diffusion models like Stable Diffusion.
    /// </remarks>
    private const int PIXART_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    /// <remarks>
    /// The VAE spatially downsamples by a factor of 8, meaning a 1024x1024 image
    /// becomes a 128x128 latent representation. This reduces computational cost during diffusion.
    /// </remarks>
    private const int PIXART_VAE_SCALE_FACTOR = 8;

    #endregion

    #region Fields

    /// <summary>
    /// The DiT-based noise predictor.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The conditioning module (T5-style text encoder).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Model size variant (alpha, sigma, delta).
    /// </summary>
    private readonly string _modelSize;

    /// <summary>
    /// Hidden dimension for the transformer.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Number of transformer blocks.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Default resolution for this model variant.
    /// </summary>
    private readonly int _defaultResolution;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => PIXART_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    public string ModelSize => _modelSize;

    /// <summary>
    /// Gets the hidden dimension of the transformer.
    /// </summary>
    public int HiddenDimension => _hiddenDim;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads => _numHeads;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the default resolution for this model.
    /// </summary>
    public int DefaultResolution => _defaultResolution;

    /// <summary>
    /// Gets whether this model supports variable aspect ratios.
    /// </summary>
    public bool SupportsVariableAspectRatio => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PixArtModel with default parameters.
    /// </summary>
    /// <remarks>
    /// Creates a PixArt-α model with:
    /// - 1024x1024 default resolution
    /// - 1152 hidden dimension
    /// - 16 attention heads
    /// - 28 transformer layers
    /// </remarks>
    /// <summary>
    /// Initializes a new instance of PixArtModel with full customization support.
    /// </summary>
    /// <param name="modelSize">Model variant: "alpha" (1024px), "sigma" (512px), or "delta" (256px).</param>
    /// <param name="conditioner">Optional conditioning module for text encoding.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PixArtModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        string modelSize = DefaultModelSize,
        IConditioningModule<T>? conditioner = null,
        INoiseScheduler<T>? scheduler = null,
        int? seed = null)
        : base(CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler(seed), architecture)
    {
        _modelSize = modelSize.ToLowerInvariant();
        _conditioner = conditioner;

        // Configure based on model size
        var (hiddenDim, numHeads, numLayers, resolution) = GetModelConfiguration(_modelSize);
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _defaultResolution = resolution;

        // Initialize mutable neural network layers
        InitializeLayers(seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the DiT noise predictor and VAE layers.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(int? seed)
    {
        // Create VAE
        _vae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: PIXART_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        // Create DiT noise predictor with PixArt-specific configuration
        _dit = CreateDiTPredictor(seed);
    }

    #endregion

    /// <summary>
    /// Creates the default options for PixArt-α.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000,
            BetaStart = 0.0001,
            BetaEnd = 0.02,
            BetaSchedule = BetaSchedule.SquaredCosine
        };
    }

    /// <summary>
    /// Creates the default scheduler for PixArt-α.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler(int? seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var config = new SchedulerConfig<T>(
            trainTimesteps: 1000,
            betaStart: ops.FromDouble(0.0001),
            betaEnd: ops.FromDouble(0.02),
            betaSchedule: BetaSchedule.SquaredCosine,
            predictionType: DiffusionPredictionType.Epsilon);
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Gets model configuration based on size variant.
    /// </summary>
    private static (int hiddenDim, int numHeads, int numLayers, int resolution) GetModelConfiguration(string modelSize)
    {
        return modelSize switch
        {
            "alpha" => (1152, 16, 28, 1024),  // PixArt-α: 600M params, 1024px
            "sigma" => (1152, 16, 28, 512),   // PixArt-Σ: Similar to α, 512px
            "delta" => (768, 12, 12, 256),    // PixArt-δ: Smaller variant
            "xl" => (1536, 24, 32, 1024),     // PixArt-XL: Larger variant
            _ => (1152, 16, 28, 1024)         // Default to α configuration
        };
    }

    /// <summary>
    /// Creates a DiT predictor configured for PixArt-α.
    /// </summary>
    private DiTNoisePredictor<T> CreateDiTPredictor(int? seed)
    {
        // PixArt uses DiT with cross-attention for text conditioning
        // and AdaLN-single for timestep conditioning
        return new DiTNoisePredictor<T>(
            inputChannels: PIXART_LATENT_CHANNELS,
            hiddenSize: _hiddenDim,
            numHeads: _numHeads,
            numLayers: _numLayers,
            contextDim: 4096,  // T5-XXL hidden dimension
            patchSize: 2,
            seed: seed);
    }

    #region Generation Methods

    /// <summary>
    /// Generates an image with PixArt-α's efficient DiT architecture.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt for things to avoid.</param>
    /// <param name="width">Image width (should be divisible by 8).</param>
    /// <param name="height">Image height (should be divisible by 8).</param>
    /// <param name="numInferenceSteps">Number of denoising steps (20-50 recommended).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (4.0-7.5 recommended).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>The generated image tensor.</returns>
    /// <remarks>
    /// PixArt-α typically uses fewer steps than SDXL due to its efficient architecture.
    /// A guidance scale of 4.5 is commonly used (lower than SDXL's typical 7.5).
    /// </remarks>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 1024,
        int height = 1024,
        int numInferenceSteps = 20,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Default to PixArt's recommended guidance scale
        var effectiveGuidanceScale = guidanceScale ?? 4.5;

        // Use base implementation with PixArt defaults
        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Generates multiple image variations with different seeds.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired images.</param>
    /// <param name="negativePrompt">Optional negative prompt for things to avoid.</param>
    /// <param name="count">Number of variations to generate.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="baseSeed">Optional base seed (variations will use baseSeed, baseSeed+1, etc.).</param>
    /// <returns>List of generated image tensors.</returns>
    public virtual List<Tensor<T>> GenerateVariations(
        string prompt,
        string? negativePrompt = null,
        int count = 4,
        int width = 1024,
        int height = 1024,
        int numInferenceSteps = 20,
        double? guidanceScale = null,
        int? baseSeed = null)
    {
        var results = new List<Tensor<T>>();
        var startSeed = baseSeed ?? RandomGenerator.Next();

        for (int i = 0; i < count; i++)
        {
            var image = GenerateFromText(
                prompt,
                negativePrompt,
                width,
                height,
                numInferenceSteps,
                guidanceScale,
                startSeed + i);
            results.Add(image);
        }

        return results;
    }

    /// <summary>
    /// Generates an image with specified aspect ratio preset.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="aspectRatio">Aspect ratio preset (e.g., "16:9", "4:3", "1:1", "9:16").</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="baseResolution">Base resolution for calculation (default 1024).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The generated image tensor.</returns>
    public virtual Tensor<T> GenerateWithAspectRatio(
        string prompt,
        string aspectRatio = "1:1",
        string? negativePrompt = null,
        int baseResolution = 1024,
        int numInferenceSteps = 20,
        double? guidanceScale = null,
        int? seed = null)
    {
        var (width, height) = CalculateDimensionsForAspectRatio(aspectRatio, baseResolution);

        return GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            guidanceScale,
            seed);
    }

    /// <summary>
    /// Calculates width and height for a given aspect ratio.
    /// </summary>
    private static (int width, int height) CalculateDimensionsForAspectRatio(string aspectRatio, int baseResolution)
    {
        var parts = aspectRatio.Split(':');
        if (parts.Length != 2 ||
            !int.TryParse(parts[0], out var widthRatio) ||
            !int.TryParse(parts[1], out var heightRatio))
        {
            return (baseResolution, baseResolution);
        }

        // Calculate dimensions that maintain aspect ratio and total pixel count ~= baseResolution^2
        var totalPixels = (double)baseResolution * baseResolution;
        var aspectFactor = (double)widthRatio / heightRatio;

        // height = sqrt(totalPixels / aspectFactor)
        var height = (int)Math.Sqrt(totalPixels / aspectFactor);
        var width = (int)(height * aspectFactor);

        // Round to nearest multiple of 8 (required for VAE)
        width = (width / 8) * 8;
        height = (height / 8) * 8;

        // Ensure minimum dimensions
        width = Math.Max(width, 256);
        height = Math.Max(height, 256);

        return (width, height);
    }

    /// <summary>
    /// Performs image-to-image transformation with PixArt-α.
    /// </summary>
    /// <param name="inputImage">The source image to transform.</param>
    /// <param name="prompt">The text prompt for the transformation.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">How much to transform (0.0 = keep original, 1.0 = full generation).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The transformed image tensor.</returns>
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.8,
        int numInferenceSteps = 20,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Default to PixArt's recommended guidance scale
        var effectiveGuidanceScale = guidanceScale ?? 4.5;

        return base.ImageToImage(
            inputImage,
            prompt,
            negativePrompt,
            strength,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Gets the recommended settings for this model variant.
    /// </summary>
    /// <returns>A tuple containing (inferenceSteps, guidanceScale, resolution).</returns>
    public (int inferenceSteps, double guidanceScale, int resolution) GetRecommendedSettings()
    {
        return _modelSize switch
        {
            "alpha" => (20, 4.5, 1024),
            "sigma" => (20, 4.5, 512),
            "delta" => (15, 4.0, 256),
            "xl" => (25, 5.0, 1024),
            _ => (20, 4.5, _defaultResolution)
        };
    }

    /// <summary>
    /// Gets supported resolutions for this model variant.
    /// </summary>
    /// <returns>List of supported resolution presets.</returns>
    public List<(int width, int height, string name)> GetSupportedResolutions()
    {
        var baseRes = _defaultResolution;
        return new List<(int, int, string)>
        {
            (baseRes, baseRes, "Square 1:1"),
            (baseRes, (int)(baseRes * 0.75), "Landscape 4:3"),
            ((int)(baseRes * 0.75), baseRes, "Portrait 3:4"),
            ((int)(baseRes * 1.33), (int)(baseRes * 0.75), "Wide 16:9"),
            ((int)(baseRes * 0.75), (int)(baseRes * 1.33), "Tall 9:16"),
            ((int)(baseRes * 1.5), (int)(baseRes * 0.625), "Cinematic 2.4:1"),
        };
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _dit.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _dit.ParameterCount)
        {
            throw new ArgumentException(
                $"Parameter count mismatch. Expected {_dit.ParameterCount} but received {parameters.Length}.",
                nameof(parameters));
        }

        _dit.SetParameters(parameters);
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
        var clone = new PixArtModel<T>(
            modelSize: _modelSize,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());

        // Copy parameters
        clone.SetParameters(GetParameters());

        return clone;
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"PixArt-{_modelSize}",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = $"PixArt-{_modelSize} text-to-image diffusion model using DiT architecture",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-xl-2");
        metadata.SetProperty("base_model", "PixArt-alpha");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("context_dim", 4096);
        metadata.SetProperty("hidden_dim", _hiddenDim);
        metadata.SetProperty("num_heads", _numHeads);
        metadata.SetProperty("num_layers", _numLayers);
        metadata.SetProperty("default_resolution", _defaultResolution);
        metadata.SetProperty("vae_scale_factor", PIXART_VAE_SCALE_FACTOR);
        metadata.SetProperty("latent_channels", PIXART_LATENT_CHANNELS);

        return metadata;
    }

    #endregion
}

/// <summary>
/// Options for PixArt-α model configuration.
/// </summary>
public class PixArtOptions<T> : DiffusionModelOptions<T>
{
    /// <summary>
    /// Model size variant. Default: "alpha".
    /// </summary>
    /// <remarks>
    /// Available variants:
    /// - "alpha": Full 1024px model (600M params)
    /// - "sigma": 512px variant
    /// - "delta": Smaller 256px variant
    /// - "xl": Larger variant with more parameters
    /// </remarks>
    public string ModelSize { get; set; } = PixArtModel<T>.DefaultModelSize;

    /// <summary>
    /// Whether to use T5-XXL text encoder. Default: true.
    /// </summary>
    /// <remarks>
    /// T5-XXL provides better text understanding but requires more memory.
    /// Disable for faster inference with slightly reduced prompt adherence.
    /// </remarks>
    public bool UseT5Encoder { get; set; } = true;

    /// <summary>
    /// Text encoder variant to use. Default: "xxl".
    /// </summary>
    /// <remarks>
    /// Options:
    /// - "xxl": T5-XXL (4.3B params, best quality)
    /// - "xl": T5-XL (3B params, faster)
    /// - "large": T5-Large (770M params, much faster)
    /// </remarks>
    public string TextEncoderSize { get; set; } = "xxl";

    /// <summary>
    /// Default guidance scale for PixArt. Default: 4.5.
    /// </summary>
    /// <remarks>
    /// PixArt typically works well with lower guidance scales than SDXL.
    /// Range: 3.0-7.5 (4.5 is commonly used).
    /// </remarks>
    public double DefaultGuidanceScale { get; set; } = 4.5;
}
