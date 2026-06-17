using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.Panorama;

/// <summary>
/// MultiDiffusion model for generating seamless panoramic and ultra-wide images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MultiDiffusion generates arbitrarily wide/tall images by running overlapping diffusion
/// passes and averaging the denoised results in the overlap regions. This produces seamless
/// panoramas without visible seam artifacts.
/// </para>
/// <para>
/// <b>For Beginners:</b> MultiDiffusion creates panoramic images wider than what the model
/// normally generates. It works by generating overlapping patches and blending them together
/// seamlessly, like taking multiple photos and stitching them into a panorama.
/// </para>
/// <para>
/// Reference: Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", ICML 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 2048, NumInferenceSteps = 30 };
/// var model = new MultiDiffusionModel&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 256 });
/// var panorama = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", "https://arxiv.org/abs/2302.08113", Year = 2023, Authors = "Bar-Tal et al.")]
public class MultiDiffusionModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override long ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public MultiDiffusionModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        // Propagate `seed` into the options so the base RandomGenerator (and thus the
        // training-timestep sampling in DiffusionModelBase.Train) is deterministic;
        // otherwise it falls back to a non-deterministic secure RNG and the training
        // invariants become flaky (see SpotDiffusionModel for the detailed rationale).
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear, Seed = seed },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 4, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }

    public override Vector<T> GetParameters()
    {
        var pp = _predictor.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(pp.Length + vp.Length);
        for (int i = 0; i < pp.Length; i++) combined[i] = pp[i];
        for (int i = 0; i < vp.Length; i++) combined[pp.Length + i] = vp[i];
        return combined;
    }

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
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    public override IDiffusionModel<T> Clone()
    {
        // Clone the ACTUAL predictor and VAE — the previous code passed neither, so the new instance
        // rebuilt InitializeLayers' DEFAULT foundation-scale UNet (320 base channels, ~643 M params)
        // while this model may hold a small/custom predictor (e.g. a test-scale UNet, ~10 M params).
        // GetParameters() then returned the source's param count and clone.SetParameters threw
        // "Expected 643774499 parameters, got 10342915". Passing the cloned predictor/VAE (and the
        // same architecture/options/scheduler) makes the clone structurally identical to the source.
        var clone = new MultiDiffusionModel<T>(
            architecture: Architecture,
            options: Options as DiffusionModelOptions<T>,
            scheduler: Scheduler,
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            seed: RandomGenerator.Next());
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "MultiDiffusion", Version = "1.0",
            Description = "Seamless panoramic generation via overlapping diffusion path averaging",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "overlapping-patch-panorama");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
