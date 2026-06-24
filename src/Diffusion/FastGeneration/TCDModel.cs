using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// Trajectory Consistency Distillation (TCD) model for high-quality few-step generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TCD extends LCM by using a trajectory-aware loss that considers the entire denoising
/// path, not just individual timestep consistency. Combined with a stochastic noise
/// injection strategy during inference, TCD produces higher-quality images than LCM
/// at the same step count, especially at 2-4 steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> TCD improves upon LCM by being smarter about how it trains.
/// While LCM focuses on individual steps, TCD considers the entire journey from noise
/// to image. Adding a tiny bit of random noise during generation also helps — like how
/// a slight shake of the camera can sometimes produce a more natural photo.
/// </para>
/// <para>
/// Reference: Zheng et al., "Trajectory Consistency Distillation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 512, NumInferenceSteps = 4 };
/// var model = new TCDModel&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Trajectory Consistency Distillation", "https://arxiv.org/abs/2402.19159", Year = 2024, Authors = "Zheng et al.")]
public class TCDModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 0.0;

    /// <summary>
    /// Paper-optimal sampling step count. TCD is a trajectory-consistency
    /// distillation model designed for high-quality few-step generation; the
    /// authors report 4 steps as the sweet spot (8 max recommended). The base
    /// <see cref="DiffusionModelOptions{T}.DefaultInferenceSteps"/> is the
    /// generic 10-step DDIM default, which both wastes ~2.5× the compute TCD
    /// needs and is not how the distilled model is meant to be sampled — so the
    /// ctor pins this value, matching every other few-step model in this folder
    /// (e.g. ConsistencyModel = 2, FluxSchnell = 4).
    /// </summary>
    private const int OPTIMAL_INFERENCE_STEPS = 4;

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
    public override long ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Initializes a new TCD model.
    /// </summary>
    public TCDModel(
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
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear,
                DefaultInferenceSteps = OPTIMAL_INFERENCE_STEPS
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
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: 768, architecture: Architecture, seed: seed);

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
        int pc = checked((int)_predictor.ParameterCount);
        int vc = checked((int)_vae.ParameterCount);
        long expectedTotal = (long)pc + vc;
        if (parameters.Length != expectedTotal)
            throw new ArgumentException($"Expected {expectedTotal} parameters, got {parameters.Length}.", nameof(parameters));
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
        var clone = new TCDModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Trajectory Consistency Distillation (TCD)", Version = "1.0",
            Description = "Trajectory-aware consistency distillation with stochastic noise injection for high-quality few-step generation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "trajectory-consistency-unet");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("distillation_method", "trajectory-consistency");
        m.SetProperty("optimal_steps", OPTIMAL_INFERENCE_STEPS);
        m.SetProperty("max_recommended_steps", 8);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
