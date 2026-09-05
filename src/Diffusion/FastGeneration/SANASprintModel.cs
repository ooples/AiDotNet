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
/// SANA Sprint model for ultra-fast 1-step generation from the SANA architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SANA Sprint is the distilled version of SANA that generates 1024x1024 images in a
/// single step. Uses the efficient linear attention DiT backbone from SANA combined
/// with hybrid distillation (consistency + adversarial) for real-time generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> SANA is already fast due to its efficient architecture.
/// SANA Sprint makes it even faster — single-step generation at 1024x1024 resolution.
/// This makes it one of the fastest high-resolution generators available, suitable for
/// real-time applications like interactive image editing.
/// </para>
/// <para>
/// Reference: NVIDIA, "SANA Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation", 2025
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 32, DefaultInferenceSteps = 1 };
/// var model = new SANASprintModel&lt;float&gt;(options: options);
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 32, 32, 32 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SANA Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation", "https://arxiv.org/abs/2503.09641", Year = 2025, Authors = "NVIDIA")]
public partial class SANASprintModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 32;
    private const double DEFAULT_GUIDANCE = 0.0;

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

    /// <summary>
    /// Initializes a new SANA Sprint model.
    /// </summary>
    public SANASprintModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        SiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear,
                // SANA-Sprint paper (Xie et al. 2025) generates in a single step.
                DefaultInferenceSteps = 1
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(SiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 2048,
            numLayers: 20,
            numHeads: 32,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.41407, seed: seed);
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "SANA Sprint", Version = "1.0",
            Description = "Ultra-fast single-step 1024x1024 generation via SANA architecture with hybrid distillation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "sana-distilled-linear-dit");
        m.SetProperty("base_model", "SANA");
        m.SetProperty("text_encoder", "Gemma-2B");
        m.SetProperty("distillation_method", "hybrid-consistency-adversarial");
        m.SetProperty("optimal_steps", 1);
        m.SetProperty("max_recommended_steps", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
