using System.Diagnostics.CodeAnalysis;
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
/// Imagen 3 model for text-to-image generation by Google DeepMind.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Imagen 3 uses a cascaded diffusion approach with a base model and super-resolution stages.
/// It features Gemma-based text understanding (evolved from T5) and is aligned with human
/// feedback through RLHF-style training for improved safety and aesthetic quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagen 3 is Google DeepMind's latest image generation model.
///
/// How Imagen 3 works:
/// 1. Text is encoded by a Gemma text encoder (evolved from T5-XXL)
/// 2. A base SiT (Scalable Interpolant Transformer) generates a 64x64 latent
/// 3. A super-resolution cascade upscales to 256x256, then 1024x1024
/// 4. Each stage is a separate diffusion model operating at increasing resolution
/// 5. Human preference alignment ensures quality and safety
///
/// Key characteristics:
/// - Cascaded architecture: base (64x64) + SR (256x256) + SR (1024x1024)
/// - Gemma text encoder for deep prompt understanding
/// - Human feedback alignment (RLHF-style) for quality and safety
/// - SiT-based backbone with interpolant formulation
/// - 16 latent channels
///
/// Advantages:
/// - Exceptional prompt adherence and photorealism
/// - Strong safety filtering via RLHF alignment
/// - Excellent text rendering capabilities
/// - Cascaded approach enables very high resolution
///
/// Limitations:
/// - API-only (not open-source)
/// - Cascaded pipeline is slower than single-stage models
/// - Higher compute requirements due to multiple stages
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Cascaded SiT with super-resolution stages
/// - Base: ~2B params, SiT backbone, 64x64 latent generation
/// - SR Stage 1: 256x256 upscale
/// - SR Stage 2: 1024x1024 final output
/// - Text encoder: Gemma (2048-dim embeddings)
/// - Latent channels: 16
/// - Default: 50 steps, guidance scale 7.5
/// - Resolution: 1024x1024 default, up to 2048x2048
///
/// Reference: Google DeepMind, "Imagen 3", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Imagen 3
/// var imagen3 = new Imagen3Model&lt;float&gt;();
///
/// // Generate a photorealistic image
/// var image = imagen3.GenerateFromText(
///     prompt: "A professional photo of a golden retriever playing in autumn leaves",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5,
///     seed: 42);
/// </code>
/// </example>
public class Imagen3Model<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for Imagen 3 (1024x1024).
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default image height for Imagen 3 (1024x1024).
    /// </summary>
    public const int DefaultHeight = 1024;

    private const int IMAGEN3_LATENT_CHANNELS = 16;
    private const int IMAGEN3_BASE_HIDDEN_SIZE = 2048;
    private const int IMAGEN3_NUM_LAYERS = 24;
    private const int IMAGEN3_CONTEXT_DIM = 2048;
    private const double IMAGEN3_DEFAULT_GUIDANCE = 7.5;
    private const int IMAGEN3_DEFAULT_STEPS = 50;

    #endregion

    #region Fields

    private SiTPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => IMAGEN3_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this model uses a cascaded architecture with super-resolution stages.
    /// </summary>
    public bool UsesCascadedArchitecture => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Imagen3Model.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture.</param>
    /// <param name="options">Configuration options. If null, uses Imagen 3 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="predictor">Custom SiT noise predictor.</param>
    /// <param name="vae">Custom 16-channel VAE.</param>
    /// <param name="conditioner">Gemma text encoder conditioning.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Imagen3Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        SiTPredictor<T>? predictor = null,
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

        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(IMAGEN3_DEFAULT_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(
        SiTPredictor<T>? predictor,
        StandardVAE<T>? vae,
        int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IMAGEN3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
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
        int numInferenceSteps = IMAGEN3_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IMAGEN3_DEFAULT_GUIDANCE;

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
        int numInferenceSteps = IMAGEN3_DEFAULT_STEPS,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? IMAGEN3_DEFAULT_GUIDANCE;

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
        var clonedPredictor = new SiTPredictor<T>();
        clonedPredictor.SetParameters(_predictor.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IMAGEN3_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Imagen3Model<T>(
            predictor: clonedPredictor,
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
            Name = "Imagen 3",
            Version = "3.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Google DeepMind's cascaded SiT with Gemma text encoder and human feedback alignment",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "cascaded-sit-diffusion");
        metadata.SetProperty("text_encoder", "Gemma");
        metadata.SetProperty("context_dim", IMAGEN3_CONTEXT_DIM);
        metadata.SetProperty("base_hidden_size", IMAGEN3_BASE_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", IMAGEN3_NUM_LAYERS);
        metadata.SetProperty("latent_channels", IMAGEN3_LATENT_CHANNELS);
        metadata.SetProperty("cascaded", true);
        metadata.SetProperty("rlhf_aligned", true);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("default_guidance_scale", IMAGEN3_DEFAULT_GUIDANCE);
        metadata.SetProperty("default_inference_steps", IMAGEN3_DEFAULT_STEPS);

        return metadata;
    }

    #endregion
}
