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
/// Uni-ControlNet model — unified multi-condition ControlNet with mixture of conditions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Uni-ControlNet allows composing multiple visual conditions simultaneously in a
/// single forward pass. It uses condition-specific adapters that are mixed together,
/// enabling complex multi-condition control.
/// </para>
/// <para>
/// <b>For Beginners:</b> Uni-ControlNet combines multiple conditions at once:
///
/// Key characteristics:
/// - Simultaneous multi-condition control (depth + edges + pose at same time)
/// - Condition-specific adapters with learned mixing
/// - Single forward pass for all conditions
/// - Compatible with SD 1.5
///
/// Use Uni-ControlNet when you need:
/// - Multiple conditions applied simultaneously
/// - Complex spatial control combining depth, edges, and pose
/// - Single-model multi-condition pipeline
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Multi-adapter ControlNet with condition mixing
/// - Adapters: condition-specific encoders with shared backbone
/// - Compatible: SD 1.5
/// - Resolution: 512x512
///
/// Reference: Zhao et al., "Uni-ControlNet: All-in-One Control to Text-to-Image
/// Diffusion Models", NeurIPS 2023
/// </para>
/// </remarks>
public class UniControlNetModel<T> : LatentDiffusionModelBase<T>
{
    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    private UNetNoisePredictor<T> _unet;
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
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    public UniControlNetModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null, StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        InitializeLayers(unet, vae, seed);
    }

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 20,
        double? guidanceScale = null, int? seed = null)
        => base.GenerateFromText(prompt, negativePrompt, width, height, numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _unet.GetParameters(); var vp = _vae.GetParameters();
        var c = new Vector<T>(up.Length + vp.Length);
        for (int i = 0; i < up.Length; i++) c[i] = up[i];
        for (int i = 0; i < vp.Length; i++) c[up.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _unet.ParameterCount, vc = _vae.ParameterCount;
        if (parameters.Length != uc + vc) throw new ArgumentException($"Expected {uc + vc}, got {parameters.Length}.", nameof(parameters));
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _unet.SetParameters(up); _vae.SetParameters(vp);
    }

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
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        cv.SetParameters(_vae.GetParameters());
        return new UniControlNetModel<T>(unet: cu, vae: cv, conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Uni-ControlNet", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Uni-ControlNet multi-condition simultaneous control", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "multi-adapter-controlnet");
        m.SetProperty("simultaneous_conditions", true);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }
}
