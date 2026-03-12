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

namespace AiDotNet.Diffusion.SuperResolution;

/// <summary>
/// TSD-SR: Timestep-Shifted Diffusion for fast and high-quality super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TSD-SR shifts the diffusion timestep range for SR tasks, recognizing that SR needs
/// less noise removal than generation from scratch. By starting from lower noise levels
/// and using an adapted schedule, it achieves high-quality 4x upscaling in just 4-10
/// diffusion steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> For super-resolution, you don't need to start from pure noise
/// — you already have the low-res image. TSD-SR is smart about this: it only adds a
/// little noise and removes it in fewer steps, making SR 5-10x faster than using a
/// standard diffusion model for upscaling.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD2.1 U-Net with timestep-shifted schedule
/// - Text encoder: OpenCLIP ViT-H/14 (1024-dim)
/// - Timestep shifting: Starts from lower noise levels (t_max ~ 200-400 vs 1000)
/// - Optimal steps: 4-10 (vs 50+ for standard diffusion SR)
/// - Speed improvement: 5-10x faster than standard diffusion SR
/// - Scale factor: 4x upscaling
/// </para>
/// </remarks>
public class TSDSRModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int TSDSR_CONTEXT_DIM = 1024;
    private const double DEFAULT_GUIDANCE = 4.0;
    private const int OPTIMAL_STEPS = 4;
    private const int SCALE_FACTOR = 4;

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

    public TSDSRModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
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
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS + 4, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: TSDSR_CONTEXT_DIM, architecture: Architecture, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
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
        var clone = new TSDSRModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "TSD-SR", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Timestep-shifted diffusion for fast 4-10 step super-resolution",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "timestep-shifted-sd21-sr-unet");
        m.SetProperty("base_model", "Stable Diffusion 2.1");
        m.SetProperty("text_encoder", "OpenCLIP ViT-H/14");
        m.SetProperty("context_dim", TSDSR_CONTEXT_DIM);
        m.SetProperty("optimal_steps", OPTIMAL_STEPS);
        m.SetProperty("scale_factor", SCALE_FACTOR);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("speedup", "5-10x vs standard diffusion SR");
        return m;
    }
}
