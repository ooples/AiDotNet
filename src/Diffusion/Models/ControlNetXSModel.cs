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
/// ControlNet-XS model — lightweight control network with minimal parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet-XS is a significantly smaller variant of ControlNet that achieves comparable
/// control quality with only ~1% of the original ControlNet parameters. It uses a
/// streamlined encoder that directly injects control signals into the base model.
/// </para>
/// <para>
/// <b>For Beginners:</b> ControlNet-XS is ControlNet but much smaller and faster:
///
/// Key differences from ControlNet:
/// - 50-100x fewer parameters than standard ControlNet
/// - Faster inference with minimal quality loss
/// - Simpler architecture: thin control encoder
/// - Works with SD 1.5 and SDXL
///
/// Use ControlNet-XS when you need:
/// - Spatial control with minimal compute overhead
/// - Edge/depth/pose-guided generation
/// - Mobile or edge deployment scenarios
/// </para>
/// <para>
/// Technical specifications:
/// - Control encoder: lightweight copy with ~1% of base model parameters
/// - Compatible: SD 1.5, SD 2.x, SDXL
/// - Conditions: depth, canny edges, segmentation, pose
/// - Injection: direct feature addition (no zero convolutions)
///
/// Reference: Zavadski et al., "ControlNet-XS", 2024
/// </para>
/// </remarks>
public class ControlNetXSModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private UNetNoisePredictor<T> _controlEncoder;
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
    public override int ParameterCount => _unet.ParameterCount + _controlEncoder.ParameterCount + _vae.ParameterCount;

    /// <summary>Gets the lightweight control encoder.</summary>
    public UNetNoisePredictor<T> ControlEncoder => _controlEncoder;

    #endregion

    #region Constructor

    public ControlNetXSModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        UNetNoisePredictor<T>? controlEncoder = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        InitializeLayers(unet, controlEncoder, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_controlEncoder), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, UNetNoisePredictor<T>? controlEncoder,
        StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);

        // Lightweight control encoder (~1% of main U-Net)
        _controlEncoder = controlEncoder ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [4],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = 20, double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _unet.GetParameters();
        var cp = _controlEncoder.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(up.Length + cp.Length + vp.Length);
        int offset = 0;
        for (int i = 0; i < up.Length; i++) combined[offset + i] = up[i]; offset += up.Length;
        for (int i = 0; i < cp.Length; i++) combined[offset + i] = cp[i]; offset += cp.Length;
        for (int i = 0; i < vp.Length; i++) combined[offset + i] = vp[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _unet.ParameterCount, cc = _controlEncoder.ParameterCount, vc = _vae.ParameterCount;
        if (parameters.Length != uc + cc + vc)
            throw new ArgumentException($"Expected {uc + cc + vc}, got {parameters.Length}.", nameof(parameters));

        var up = new Vector<T>(uc); var cp = new Vector<T>(cc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < cc; i++) cp[i] = parameters[uc + i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + cc + i];
        _unet.SetParameters(up); _controlEncoder.SetParameters(cp); _vae.SetParameters(vp);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cu = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM);
        cu.SetParameters(_unet.GetParameters());
        var cc = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [4],
            contextDim: CROSS_ATTENTION_DIM);
        cc.SetParameters(_controlEncoder.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        cv.SetParameters(_vae.GetParameters());
        return new ControlNetXSModel<T>(unet: cu, controlEncoder: cc, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "ControlNet-XS", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "ControlNet-XS lightweight spatial control with ~1% parameters",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "controlnet-xs");
        m.SetProperty("control_encoder_ratio", 0.01);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }

    #endregion
}
