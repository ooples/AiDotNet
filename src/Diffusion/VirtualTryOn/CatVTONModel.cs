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

namespace AiDotNet.Diffusion.VirtualTryOn;

/// <summary>
/// CatVTON model for concatenation-based virtual try-on without warping modules.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CatVTON simplifies virtual try-on by concatenating garment features directly with
/// person features as input, eliminating the need for explicit warping modules. The diffusion
/// model learns to implicitly handle deformation and blending.
/// </para>
/// <para>
/// <b>For Beginners:</b> CatVTON is a simpler approach to virtual try-on. Instead of
/// complex warping to fit clothes to a body, it simply feeds both the person and garment
/// images together and lets the AI figure out how to combine them naturally.
/// </para>
/// <para>
/// Reference: Chong et al., "CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 384, NumInferenceSteps = 30 };
/// var model = new CatVTONModel&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 48 });
/// var tryOn = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ImageEditing)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models", "https://arxiv.org/abs/2407.15886", Year = 2024, Authors = "Chong et al.")]
public class CatVTONModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 2.5;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override long ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public CatVTONModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 12, outputChannels: LATENT_CHANNELS,
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
        // Delegate to the predictor/VAE's own Clone implementations (mirrors
        // RealESRGAN / SeedEdit3 / ImprovedConsistencyModel, PR #1555/#1565).
        // The previous body constructed a fresh CatVTONModel — which rebuilds
        // the DEFAULT SD-1.5-scale predictor/VAE — and then SetParameters with
        // this model's weights: for a model constructed with custom-sized
        // components the parameter counts differ and SetParameters throws, and
        // even at default scale the fresh components re-hit the lazy-init
        // divergence bug.
        var clone = new CatVTONModel<T>(
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner, seed: RandomGenerator.Next());
        return clone;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "CatVTON", Version = "1.0",
            Description = "Concatenation-based virtual try-on without explicit warping",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "concat-diffusion-tryon");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
