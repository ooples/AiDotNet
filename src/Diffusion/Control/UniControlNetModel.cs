using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// Uni-ControlNet model for simultaneous multi-condition control with condition-specific adapters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uni-ControlNet allows composing multiple visual conditions simultaneously in a single
/// forward pass. It uses condition-specific adapters that are mixed together with learned
/// weights, enabling complex multi-condition spatial control.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)</description></item>
/// <item><description>Condition-specific encoder adapters for each control type</description></item>
/// <item><description>Learned condition mixing weights for multi-condition composition</description></item>
/// <item><description>Standard SD 1.5 VAE for image encoding/decoding</description></item>
/// <item><description>DDIM scheduler for efficient inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Uni-ControlNet combines multiple spatial conditions simultaneously.
///
/// How Uni-ControlNet works:
/// 1. Multiple condition images are provided (e.g., depth + edges + pose simultaneously)
/// 2. Each condition is processed by its specific adapter encoder
/// 3. Adapter features are mixed with learned weights for each condition
/// 4. Combined control features are injected into the SD 1.5 U-Net
/// 5. The U-Net generates an image guided by all conditions simultaneously
///
/// Key characteristics:
/// - Simultaneous multi-condition control (depth + edges + pose at same time)
/// - Condition-specific adapters with learned mixing weights
/// - Single forward pass for all conditions combined
/// - Compatible with SD 1.5 backbone
/// - No need to manually balance multiple ControlNets
///
/// When to use Uni-ControlNet:
/// - Multiple conditions applied simultaneously
/// - Complex spatial control combining depth, edges, and pose
/// - Single-model multi-condition pipeline
/// - When per-condition ControlNet loading is impractical
///
/// Limitations:
/// - SD 1.5 resolution (512x512)
/// - Adding new condition types requires retraining
/// - Condition interactions can produce unexpected results
/// - Not as flexible as separate ControlNets for per-condition tuning
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Multi-adapter ControlNet with condition mixing
/// - Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
/// - Cross-attention: 768-dim (CLIP ViT-L/14)
/// - Adapters: condition-specific encoders with shared backbone
/// - Default resolution: 512x512
/// - Scheduler: DDIM
///
/// Reference: Zhao et al., "Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models", NeurIPS 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var uniControlNet = new UniControlNetModel&lt;float&gt;();
/// var image = uniControlNet.GenerateFromText(
///     prompt: "A photorealistic cityscape at sunset",
///     width: 512, height: 512,
///     numInferenceSteps: 20,
///     guidanceScale: 7.5);
/// </code>
/// </example>
public class UniControlNetModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;
    private const int BASE_CHANNELS = 320;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

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

    public UniControlNetModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS, baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 }, contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 20,
        double? guidanceScale = null, int? seed = null)
        => base.GenerateFromText(prompt, negativePrompt, width, height, numInferenceSteps,
            guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);
        for (int i = 0; i < unetParams.Length; i++) combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++) combined[unetParams.Length + i] = vaeParams[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;
        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.", nameof(parameters));
        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < unetCount; i++) unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++) vaeParams[i] = parameters[unetCount + i];
        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());
        var clonedVae = new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());
        return new UniControlNetModel<T>(unet: clonedUnet, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Uni-ControlNet", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Uni-ControlNet multi-condition simultaneous control with learned mixing",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "multi-adapter-controlnet");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("simultaneous_conditions", true);
        metadata.SetProperty("learned_mixing", true);
        metadata.SetProperty("base_model", "SD-1.5");
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("condition_adapters", "per-condition-specific");
        metadata.SetProperty("single_forward_pass", true);
        return metadata;
    }

    #endregion
}
