using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// Step1X-Edit model for one-step image editing using consistency distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Step1X-Edit achieves single-step image editing through consistency distillation,
/// mapping directly from the source image and edit instruction to the edited result
/// in one forward pass. Built on a DiT backbone distilled from a multi-step teacher.
/// </para>
/// <para>
/// <b>For Beginners:</b> Step1X-Edit is the fastest editing model — it makes changes
/// in a single step instead of the usual 20-50. This makes editing feel instant,
/// perfect for interactive tools where you want immediate feedback.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: DiT backbone with consistency distillation
/// - Inference: Single-step (1 NFE) — no iterative denoising
/// - Latent space: 16 channels
/// - Distilled from multi-step rectified flow teacher
/// - Guidance: 1.0 (distilled models don't need CFG)
///
/// Reference: StepFun, "Step1X-Edit", 2025
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 16, Height = 1024, Width = 1024, NumInferenceSteps = 28 };
/// var model = new Step1XEditModel&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 16, 128, 128 });
/// var edited = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Editing)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Step1X-Edit", "https://arxiv.org/abs/2504.17761", Year = 2025, Authors = "StepFun")]
public class Step1XEditModel<T> : LatentDiffusionModelBase<T>
{
    // Step1X-Edit (StepFun 2025) §3 architecture summary: 16-channel latent
    // SDXL-class VAE, MM-DiT backbone trained with rectified-flow distillation
    // for *single-step* inference at deployment time. Defaults below match
    // the paper's deployment-time inference configuration:
    //   - LATENT_CHANNELS=16 matches the SDXL/Flux-style VAE (paper §3.2).
    //   - DEFAULT_STEPS=1 matches the paper's 1-step distilled editor; the
    //     diffusion-base default of 50 inference steps would waste 49 forward
    //     passes through a model that was trained to denoise in one shot.
    //   - DEFAULT_GUIDANCE=1.0 matches the paper's guidance-free single-step
    //     inference (CFG only used at training time).
    private const int LATENT_CHANNELS = 16;
    private const double DEFAULT_GUIDANCE = 1.0;
    private const int DEFAULT_STEPS = 1;

    private SiTPredictor<T> _predictor;
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

    public Step1XEditModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, SiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> {
                TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear,
                DefaultInferenceSteps = DEFAULT_STEPS },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(SiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        // The noise predictor operates on the VAE's latent representation, not
        // raw RGB pixels. Default SiTPredictor.inputChannels=4 was a mismatch
        // for Step1X (LATENT_CHANNELS=16) and made LatentDiffusionModelBase's
        // CanonicalizeGenShape rewrite the user's [B, 16, H, W] latent input
        // into [B, 4, H, W], producing output that no longer matched the input
        // shape and breaking OutputShape_ShouldMatchInputShape.
        _predictor = predictor ?? new SiTPredictor<T>(inputChannels: LATENT_CHANNELS, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305, seed: seed);
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
        var clone = new Step1XEditModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Step1X-Edit", Version = "1.0",
            Description = "Single-step image editing via consistency distillation from rectified flow teacher",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "consistency-distilled-dit-editing");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_steps", DEFAULT_STEPS);
        m.SetProperty("editing_method", "consistency-distillation");
        m.SetProperty("training", "consistency-distilled-rectified-flow");
        return m;
    }
}
