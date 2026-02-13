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
/// PixArt-Sigma model for high-resolution text-to-image generation with improved quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PixArt-Sigma is the successor to PixArt-Alpha, featuring improved training data quality,
/// higher native resolution support (up to 4K), and better aesthetic quality. It maintains
/// the efficient DiT architecture while achieving quality comparable to SDXL and DALL-E 3.
/// </para>
/// <para>
/// <b>For Beginners:</b> PixArt-Sigma is an upgraded version of PixArt-Alpha:
///
/// Key improvements over PixArt-Alpha:
/// - Higher native resolution: supports up to 4096x4096
/// - Better image quality from improved training data
/// - Flexible aspect ratios via bucket training
/// - Still very efficient (DiT-based, much faster than U-Net models)
///
/// How PixArt-Sigma works:
/// 1. Text goes through T5-XXL encoder (4096-dim)
/// 2. DiT transformer blocks denoise the latent with cross-attention to text
/// 3. Trained on high-quality curated datasets with better captions
/// 4. VAE decodes to final high-resolution image
///
/// Use PixArt-Sigma when you need:
/// - High-resolution output (up to 4K)
/// - Fast generation on limited hardware
/// - Good quality with flexible aspect ratios
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT (Diffusion Transformer) with T5-XXL text encoder
/// - Parameters: ~600M (DiT-XL/2)
/// - Text encoder: T5-XXL (4096-dim embeddings)
/// - Native resolution: up to 4096x4096
/// - Latent space: 4 channels, 8x downsampling
/// - Scheduler: DPM-Solver with 20 steps recommended
///
/// Reference: Chen et al., "PixArt-Sigma: Weak-to-Strong Training of Diffusion Transformer
/// for 4K Text-to-Image Generation", ECCV 2024
/// </para>
/// </remarks>
public class PixArtSigmaModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 4096;
    private const double DEFAULT_GUIDANCE_SCALE = 4.5;

    #endregion

    #region Fields

    private DiTNoisePredictor<T> _dit;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of PixArtSigmaModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses PixArt-Sigma defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, creates a DPM++ 2M scheduler.</param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the standard DiT-XL/2.</param>
    /// <param name="vae">Custom VAE. If null, creates the standard SDXL VAE.</param>
    /// <param name="conditioner">Text encoder conditioning module.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PixArtSigmaModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
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
            scheduler ?? new DPMSolverMultistepScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        InitializeLayers(dit, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_dit), nameof(_vae))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        StandardVAE<T>? vae,
        int? seed)
    {
        // DiT-XL/2: 28 layers, 1152 hidden, 16 heads
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 1152,
            numLayers: 28,
            numHeads: 16,
            patchSize: 2,
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
        int numInferenceSteps = 20,
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
        var ditParams = _dit.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
            combined[i] = ditParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[ditParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var ditCount = _dit.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != ditCount + vaeCount)
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
            ditParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[ditCount + i];

        _dit.SetParameters(ditParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: 1152,
            numLayers: 28, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        clonedVae.SetParameters(_vae.GetParameters());

        return new PixArtSigmaModel<T>(
            dit: clonedDit, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "PixArt-Sigma",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "PixArt-Sigma DiT-based text-to-image model with 4K resolution support",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-xl-2");
        metadata.SetProperty("text_encoder", "T5-XXL");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("max_resolution", 4096);

        return metadata;
    }

    #endregion
}
