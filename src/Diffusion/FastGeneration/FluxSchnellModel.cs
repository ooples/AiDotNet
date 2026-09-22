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
/// FLUX.1 Schnell for ultra-fast 1-4 step generation from the FLUX architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLUX.1 Schnell is the distilled variant of FLUX.1 optimized for 1-4 step generation.
/// Uses the same double-stream transformer architecture as FLUX.1 Dev but distilled for
/// speed. Requires no classifier-free guidance, producing high-quality images instantly.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLUX.1 is one of the best open-source image generators.
/// FLUX.1 Schnell (German for "fast") is its speed-optimized version that generates
/// images in just 1-4 steps. It's free for commercial use and produces remarkably good
/// images for a distilled model.
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.1 Schnell", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 16, DefaultInferenceSteps = 4 };
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 16, 128, 128 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new FluxSchnellModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var generated = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FLUX.1 Schnell", "https://blackforestlabs.ai/flux-1-schnell/", Year = 2024, Authors = "Black Forest Labs")]
public partial class FluxSchnellModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 16;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 0.0;
    private const int DEFAULT_STEPS = 4;

    private FluxDoubleStreamPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    // Resolved (never-null) construction seed for the default predictor/VAE. Clone() reuses it so a
    // never-forwarded — still lazy — predictor materializes IDENTICAL weights in the clone instead of
    // diverging on a fresh seed. Resolved once, so two separately-constructed unseeded models still differ.
    private readonly int _layerSeed;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />

    public FluxSchnellModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FluxDoubleStreamPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        // Resolve null → a concrete seed so Clone() can reconstruct identical lazy weights from it.
        _layerSeed = seed ?? RandomGenerator.Next();
        InitializeLayers(predictor, vae, _layerSeed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(FluxDoubleStreamPredictor<T>? predictor, StandardVAE<T>? vae, int seed)
    {
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(
            variant: FluxPredictorVariant.Schnell,
            inputChannels: LATENT_CHANNELS,
            contextDim: 4096,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.3611, seed: seed);
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "FLUX.1 Schnell", Version = "1.0",
            Description = "Ultra-fast 1-4 step FLUX generation, Apache 2.0 licensed",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flux-double-stream-distilled");
        m.SetProperty("base_model", "FLUX.1");
        m.SetProperty("text_encoder", "CLIP-L + T5-XXL");
        m.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        m.SetProperty("distillation_method", "guidance-distillation");
        m.SetProperty("optimal_steps", DEFAULT_STEPS);
        m.SetProperty("license", "Apache 2.0");
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
