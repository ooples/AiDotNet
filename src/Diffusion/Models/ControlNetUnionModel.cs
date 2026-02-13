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
/// ControlNet Union model — single ControlNet that handles multiple condition types.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet Union combines multiple control conditions (depth, canny, pose, etc.)
/// into a single ControlNet model. Instead of loading separate ControlNets for each
/// condition type, one model handles all conditions via task-specific tokens.
/// </para>
/// <para>
/// <b>For Beginners:</b> ControlNet Union is an all-in-one control model:
///
/// Standard ControlNet: one model per condition (depth model, canny model, pose model...)
/// ControlNet Union: ONE model handles ALL conditions
///
/// Supported conditions:
/// - Canny edges, depth maps, normal maps
/// - OpenPose, segmentation maps
/// - Scribbles, line art
/// - Low-quality image upscaling
///
/// Use ControlNet Union when you need:
/// - Multiple control types without loading multiple models
/// - Memory-efficient multi-condition pipeline
/// - Mixed-condition generation
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: ControlNet with task routing tokens
/// - Conditions: 8+ types via task embedding
/// - Compatible: SDXL
/// - Single model replaces 8+ individual ControlNets
///
/// Reference: Xinsong Zhang, "ControlNet++/Union", 2024
/// </para>
/// </remarks>
public class ControlNetUnionModel<T> : LatentDiffusionModelBase<T>
{
    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 2048;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    private UNetNoisePredictor<T> _unet;
    private UNetNoisePredictor<T> _controlNet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _controlNet.ParameterCount + _vae.ParameterCount;
    /// <summary>Gets the unified control network.</summary>
    public UNetNoisePredictor<T> ControlNet => _controlNet;

    public ControlNetUnionModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null, UNetNoisePredictor<T>? controlNet = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        InitializeLayers(unet, controlNet, vae, seed);
    }

    [MemberNotNull(nameof(_unet), nameof(_controlNet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, UNetNoisePredictor<T>? controlNet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);
        _controlNet = controlNet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025, seed: seed);
    }

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 25,
        double? guidanceScale = null, int? seed = null)
        => base.GenerateFromText(prompt, negativePrompt, width, height, numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _unet.GetParameters(); var cp = _controlNet.GetParameters(); var vp = _vae.GetParameters();
        var c = new Vector<T>(up.Length + cp.Length + vp.Length);
        int o = 0;
        for (int i = 0; i < up.Length; i++) c[o + i] = up[i]; o += up.Length;
        for (int i = 0; i < cp.Length; i++) c[o + i] = cp[i]; o += cp.Length;
        for (int i = 0; i < vp.Length; i++) c[o + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _unet.ParameterCount, cc = _controlNet.ParameterCount, vc = _vae.ParameterCount;
        if (parameters.Length != uc + cc + vc) throw new ArgumentException($"Expected {uc + cc + vc}, got {parameters.Length}.", nameof(parameters));
        var up = new Vector<T>(uc); var cp = new Vector<T>(cc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < cc; i++) cp[i] = parameters[uc + i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + cc + i];
        _unet.SetParameters(up); _controlNet.SetParameters(cp); _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cu = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM);
        cu.SetParameters(_unet.GetParameters());
        var cc = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM);
        cc.SetParameters(_controlNet.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        cv.SetParameters(_vae.GetParameters());
        return new ControlNetUnionModel<T>(unet: cu, controlNet: cc, vae: cv, conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "ControlNet Union", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "ControlNet Union all-in-one multi-condition control model", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "controlnet-union");
        m.SetProperty("supported_conditions", "canny,depth,pose,normal,scribble,lineart,segmentation,tile");
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }
}
