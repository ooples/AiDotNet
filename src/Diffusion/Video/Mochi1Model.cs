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

namespace AiDotNet.Diffusion.Video;

/// <summary>
/// Mochi 1 model for asymmetric DiT video generation with joint text-video attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mochi 1 by Genmo uses an Asymmetric Diffusion Transformer (AsymmDiT) architecture
/// with joint text-video attention and an asymmetric encoder-decoder VAE.
/// The asymmetric design uses lightweight encoding and heavy decoding for quality.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>AsymmDiT with 48 transformer layers and 3072 hidden dimension</description></item>
/// <item><description>24 attention heads with joint text-video attention (not cross-attention)</description></item>
/// <item><description>Asymmetric VAE: lightweight encoder, heavy decoder for quality</description></item>
/// <item><description>12 latent channels with 3D causal temporal compression</description></item>
/// <item><description>T5-XXL text encoder for 4096-dim context embeddings</description></item>
/// <item><description>Flow matching training objective</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Mochi 1 is a state-of-the-art open-source video generation model.
///
/// How Mochi 1 works:
/// 1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
/// 2. Video is compressed by the asymmetric VAE into 12-channel latent space
/// 3. The AsymmDiT processes text and video tokens jointly (not via cross-attention)
/// 4. Joint attention allows deep text-video interaction in every layer
/// 5. The heavy VAE decoder reconstructs high-quality video
///
/// Key characteristics:
/// - ~10B parameters (one of the largest open-source video models)
/// - Joint text-video attention (deeper integration than cross-attention)
/// - Asymmetric VAE: fast encoding, high-quality decoding
/// - 480p at 30 FPS, 84 frames (~2.8 seconds)
/// - Open-source weights under Apache 2.0 license
///
/// When to use Mochi 1:
/// - High-quality open-source video generation
/// - Research on joint attention mechanisms
/// - Text-to-video with strong motion understanding
/// - Commercial use (Apache 2.0 license)
///
/// Limitations:
/// - Very large model (10B parameters, high VRAM requirements)
/// - Limited to 480p resolution
/// - Shorter duration than some competitors
/// - Slower inference due to model size
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: AsymmDiT (Asymmetric Diffusion Transformer)
/// - Hidden dimension: 3072
/// - Transformer layers: 48
/// - Attention heads: 24
/// - Patch size: 2
/// - Latent channels: 12 (asymmetric VAE)
/// - Context dimension: 4096 (T5-XXL)
/// - Attention: Joint text-video (not cross-attention)
/// - Default: 84 frames at 30 FPS (~2.8 seconds)
/// - Training objective: Flow matching
/// - Total parameters: ~10B
/// - License: Apache 2.0
///
/// Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create with industry-standard defaults
/// var mochi = new Mochi1Model&lt;float&gt;();
///
/// // Generate video from text
/// var video = mochi.GenerateFromText(
///     prompt: "A butterfly emerging from its chrysalis in slow motion",
///     width: 848,
///     height: 480,
///     numFrames: 84,
///     fps: 30,
///     numInferenceSteps: 64,
///     guidanceScale: 4.5);
/// </code>
/// </example>
public class Mochi1Model<T> : VideoDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels from the asymmetric VAE (12).
    /// </summary>
    /// <remarks>
    /// Mochi 1's asymmetric VAE uses 12 latent channels, a middle ground between
    /// SD's 4 and HunyuanVideo's 16, optimized for the asymmetric encoder-decoder design.
    /// </remarks>
    private const int LATENT_CHANNELS = 12;

    /// <summary>
    /// Hidden dimension of the AsymmDiT transformer (3072).
    /// </summary>
    private const int HIDDEN_DIM = 3072;

    /// <summary>
    /// Number of transformer layers in the AsymmDiT (48).
    /// </summary>
    private const int NUM_LAYERS = 48;

    /// <summary>
    /// Number of attention heads (24).
    /// </summary>
    private const int NUM_HEADS = 24;

    /// <summary>
    /// Context dimension from the T5-XXL text encoder (4096).
    /// </summary>
    private const int CONTEXT_DIM = 4096;

    /// <summary>
    /// Patch size for spatiotemporal tokenization (2).
    /// </summary>
    private const int PATCH_SIZE = 2;

    /// <summary>
    /// Default number of frames (84, ~2.8 seconds at 30 FPS).
    /// </summary>
    private const int DEFAULT_NUM_FRAMES = 84;

    /// <summary>
    /// Default frames per second (30).
    /// </summary>
    private const int DEFAULT_FPS = 30;

    #endregion

    #region Fields

    /// <summary>
    /// The AsymmDiT noise predictor with joint text-video attention.
    /// </summary>
    private DiTNoisePredictor<T> _dit;

    /// <summary>
    /// The asymmetric temporal VAE (lightweight encoder, heavy decoder).
    /// </summary>
    private TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The T5-XXL text encoder conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _temporalVAE;

    /// <inheritdoc />
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsImageToVideo => false;

    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;

    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Mochi1Model with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">
    /// Diffusion model options. If null, uses Mochi 1 defaults:
    /// linear beta [0.0001, 0.02], 1000 timesteps.
    /// </param>
    /// <param name="scheduler">
    /// Noise scheduler. If null, uses flow matching scheduler matching the training objective.
    /// </param>
    /// <param name="dit">Custom DiT noise predictor. If null, creates the 48-layer AsymmDiT.</param>
    /// <param name="temporalVAE">Custom temporal VAE. If null, creates the asymmetric VAE.</param>
    /// <param name="conditioner">T5-XXL text encoder conditioning module.</param>
    /// <param name="defaultNumFrames">Default frames per generation (default: 84).</param>
    /// <param name="defaultFPS">Default frames per second (default: 30).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Mochi1Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = DEFAULT_NUM_FRAMES,
        int defaultFPS = DEFAULT_FPS,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            defaultNumFrames,
            defaultFPS,
            architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(dit, temporalVAE, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the AsymmDiT and asymmetric VAE layers using custom or default configurations.
    /// </summary>
    /// <param name="dit">Custom DiT predictor, or null for Mochi 1 defaults.</param>
    /// <param name="temporalVAE">Custom temporal VAE, or null for asymmetric VAE.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <remarks>
    /// <para>
    /// Default AsymmDiT:
    /// - 12 input/output channels, 3072 hidden dim, 48 layers, 24 heads
    /// - Patch size 2, joint text-video attention
    /// - 4096-dim context from T5-XXL
    ///
    /// Default Asymmetric VAE:
    /// - Lightweight encoder, heavy decoder
    /// - 12 latent channels, causal temporal processing
    /// - 2 temporal layers, kernel size 3
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_dit), nameof(_temporalVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        TemporalVAE<T>? temporalVAE,
        int? seed)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);

        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 2,
            temporalKernelSize: 3,
            causalMode: true,
            latentScaleFactor: 0.13025);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        return _dit.PredictNoise(latents, timestep, imageEmbedding);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();

        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
        {
            combined[i] = ditParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[ditParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var ditCount = _dit.ParameterCount;
        var vaeCount = _temporalVAE.GetParameters().Length;

        if (parameters.Length != ditCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
        {
            ditParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[ditCount + i];
        }

        _dit.SetParameters(ditParams);
        _temporalVAE.SetParameters(vaeParams);
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
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: PATCH_SIZE,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new Mochi1Model<T>(
            dit: clonedDit,
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Mochi-1",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Mochi 1 asymmetric DiT video generation with joint text-video attention",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "asymm-dit-joint-attention");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("parameters_billions", 10);
        metadata.SetProperty("training_objective", "flow-matching");
        metadata.SetProperty("open_source", true);
        metadata.SetProperty("default_frames", DEFAULT_NUM_FRAMES);

        return metadata;
    }

    #endregion
}
