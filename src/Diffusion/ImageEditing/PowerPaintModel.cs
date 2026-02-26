using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// PowerPaint v2 model for versatile task-aware image inpainting with learnable task prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PowerPaint uses learnable task prompts to handle multiple inpainting tasks including
/// text-guided object insertion, object removal, shape-guided generation, and outpainting
/// with a single model. Task-specific prompt tokens are prepended to guide behavior.
/// Built on Stable Diffusion 2.1 with OpenCLIP ViT-H text encoder.
/// </para>
/// <para>
/// <b>For Beginners:</b> PowerPaint is a versatile inpainting model that can handle
/// many different tasks: removing objects, inserting new objects, changing shapes, or
/// extending images. You just tell it which task you want, and it adapts automatically
/// through special learned task tokens.
/// </para>
/// <para>
/// Technical specifications:
/// - Base model: Stable Diffusion 2.1 (UNet)
/// - Text encoder: OpenCLIP ViT-H/14 (1024-dim context)
/// - Input channels: 9 (4 latent + 4 masked image + 1 mask)
/// - Task prompts: 4 learnable tokens per task type
/// - Tasks: text-guided insertion, removal, shape-guided, outpainting
/// - Resolution: 512x512 native
///
/// Reference: Zhuang et al., "A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting", 2024
/// </para>
/// </remarks>
public class PowerPaintModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int PP_CONTEXT_DIM = 1024;
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public PowerPaintModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 9, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1], contextDim: PP_CONTEXT_DIM, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var pp = _predictor.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(pp.Length + vp.Length);
        for (int i = 0; i < pp.Length; i++) combined[i] = pp[i];
        for (int i = 0; i < vp.Length; i++) combined[pp.Length + i] = vp[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var pc = _predictor.ParameterCount;
        var vc = _vae.ParameterCount;
        if (parameters.Length != pc + vc)
            throw new ArgumentException($"Expected {pc + vc} parameters, got {parameters.Length}.", nameof(parameters));
        var pp = new Vector<T>(pc);
        var vp = new Vector<T>(vc);
        for (int i = 0; i < pc; i++) pp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[pc + i];
        _predictor.SetParameters(pp);
        _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new PowerPaintModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "PowerPaint", Version = "2.0", ModelType = ModelType.NeuralNetwork,
            Description = "Versatile task-aware inpainting with learnable task prompts on SD2.1",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "task-prompt-sd21-inpainting");
        m.SetProperty("base_model", "Stable Diffusion 2.1");
        m.SetProperty("text_encoder", "OpenCLIP ViT-H/14");
        m.SetProperty("context_dim", PP_CONTEXT_DIM);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("tasks", "insertion, removal, shape-guided, outpainting");
        return m;
    }
}
