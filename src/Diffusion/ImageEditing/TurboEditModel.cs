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
/// TurboEdit model for fast few-step image editing using distilled SDXL Turbo.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TurboEdit adapts DDPM inversion techniques to work with few-step distilled models
/// (SDXL Turbo), enabling real-time image editing in just 3-5 steps. Uses
/// carefully calibrated noise injection to maintain edit quality at low step counts.
/// </para>
/// <para>
/// <b>For Beginners:</b> TurboEdit is fast image editing — it can modify images in
/// just 3-5 denoising steps, making it nearly instant. Great for interactive editing
/// workflows where you want to see changes in real time.
/// </para>
/// <para>
/// Technical specifications:
/// - Base model: SDXL Turbo (adversarially distilled)
/// - Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-G/14 (2048 context)
/// - Editing: Adapted DDPM inversion for few-step regime
/// - Typical steps: 3-5 (vs 20-50 for standard models)
/// - Input channels: 8 (4 latent + 4 source image latent)
/// - Guidance: Low (2.0) due to distillation
///
/// Reference: Deutch et al., "TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models", 2024
/// </para>
/// </remarks>
public class TurboEditModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int SDXL_CONTEXT_DIM = 2048;
    private const double DEFAULT_GUIDANCE = 2.0;
    private const int DEFAULT_STEPS = 4;

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

    public TurboEditModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 8, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2], contextDim: SDXL_CONTEXT_DIM, seed: seed);
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
        var clone = new TurboEditModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "TurboEdit", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Fast 3-5 step image editing with distilled SDXL Turbo inversion",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "sdxl-turbo-editing");
        m.SetProperty("base_model", "SDXL Turbo");
        m.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        m.SetProperty("text_encoder_2", "OpenCLIP ViT-G/14");
        m.SetProperty("context_dim", SDXL_CONTEXT_DIM);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_steps", DEFAULT_STEPS);
        m.SetProperty("editing_method", "adapted-ddpm-inversion");
        return m;
    }
}
