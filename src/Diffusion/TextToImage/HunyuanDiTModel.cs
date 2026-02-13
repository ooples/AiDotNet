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
/// Hunyuan-DiT model — bilingual (Chinese-English) DiT text-to-image model by Tencent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Hunyuan-DiT is Tencent's bilingual text-to-image model that uses a DiT transformer
/// backbone with dual text encoders (CLIP + multilingual T5) for both Chinese and
/// English prompt understanding.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hunyuan-DiT generates images from Chinese or English prompts:
///
/// Key characteristics:
/// - Bilingual: understands both Chinese and English prompts natively
/// - DiT backbone: transformer-based denoiser (1.5B parameters)
/// - Dual text encoders: CLIP-L/14 + mT5-XL for multilingual support
/// - Multi-resolution training with aspect ratio bucketing
/// - Human preference alignment via RLHF-like training
///
/// How Hunyuan-DiT works:
/// 1. Text goes through CLIP (visual) + mT5 (multilingual) encoders
/// 2. DiT transformer denoises with cross-attention to both embeddings
/// 3. Multi-resolution support through positional embedding interpolation
/// 4. VAE decoder produces final image
///
/// Use Hunyuan-DiT when you need:
/// - Chinese text-to-image generation
/// - Bilingual applications
/// - Open-source multilingual alternative
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT-XL with dual text conditioning
/// - Parameters: ~1.5B (DiT backbone)
/// - Text encoders: CLIP ViT-L/14 (768-dim) + mT5-XL (2048-dim)
/// - Native resolution: 1024x1024
/// - Latent space: 4 channels, 8x downsampling
/// - Guidance scale: 6.0 recommended
///
/// Reference: Li et al., "Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer
/// with Fine-Grained Chinese Understanding", 2024
/// </para>
/// </remarks>
public class HunyuanDiTModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 2048;
    private const double DEFAULT_GUIDANCE_SCALE = 6.0;

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
    /// Initializes a new instance of HunyuanDiTModel with full customization support.
    /// </summary>
    public HunyuanDiTModel(
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
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
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
        // Hunyuan-DiT-XL: 40 layers, 1408 hidden, 16 heads
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 1408,
            numLayers: 40,
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
        int numInferenceSteps = 30,
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
            inputChannels: LATENT_CHANNELS, hiddenSize: 1408,
            numLayers: 40, numHeads: 16, patchSize: 2,
            contextDim: CROSS_ATTENTION_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        clonedVae.SetParameters(_vae.GetParameters());

        return new HunyuanDiTModel<T>(
            dit: clonedDit, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Hunyuan-DiT",
            Version = "1.2",
            ModelType = ModelType.NeuralNetwork,
            Description = "Hunyuan-DiT bilingual Chinese-English DiT text-to-image model",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-xl-dual-encoder");
        metadata.SetProperty("text_encoders", "CLIP-L+mT5-XL");
        metadata.SetProperty("bilingual", true);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
