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
/// Distribution Matching Distillation v2 (DMD2) for single-step high-fidelity generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DMD2 improves upon DMD by introducing a regression loss alongside the distribution
/// matching loss, eliminating the need for an expensive GAN discriminator. Achieves
/// state-of-the-art single-step FID on ImageNet while being simpler to train than ADD.
/// </para>
/// <para>
/// <b>For Beginners:</b> DMD2 trains a student model to produce images that match the
/// distribution of a teacher (full diffusion model) in a single step. It's like having
/// a talented student who can paint a complete picture in one stroke, trained by watching
/// a master who paints in many careful strokes.
/// </para>
/// <para>
/// Reference: Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DMD2 model for single-step high-fidelity generation
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 1 };
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 64, 64 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new DMD2Model&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var generated = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Improved Distribution Matching Distillation for Fast Image Synthesis", "https://arxiv.org/abs/2405.14867", Year = 2024, Authors = "Yin et al.")]
public partial class DMD2Model<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 4;
    private const int SD15_CONTEXT_DIM = 768;
    private const double DEFAULT_GUIDANCE = 0.0;
    private const int DEFAULT_STEPS = 1;

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

    /// <summary>
    /// Initializes a new DMD2 model.
    /// </summary>
    public DMD2Model(
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
                DefaultInferenceSteps = DEFAULT_STEPS
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
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "DMD2", Version = "2.0",
            Description = "Distribution Matching Distillation v2 for single-step generation without GAN discriminator",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "distribution-matching-distillation");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", SD15_CONTEXT_DIM);
        m.SetProperty("distillation_method", "distribution-matching + regression");
        m.SetProperty("optimal_steps", DEFAULT_STEPS);
        m.SetProperty("max_recommended_steps", 4);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
