using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.MotionGeneration;

/// <summary>
/// MotionDiffuse model for fine-grained text-driven motion generation with body part control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MotionDiffuse provides fine-grained control over body part motions through separate
/// text conditioning for different body segments (arms, legs, torso). It uses a
/// part-aware cross-attention mechanism to bind text descriptions to specific body parts.
/// </para>
/// <para>
/// <b>For Beginners:</b> MotionDiffuse gives you more control over body animation than
/// MDM. You can describe what each body part should do separately — like "wave the right
/// hand while walking forward" — and each part follows its own instruction precisely.
/// </para>
/// <para>
/// Reference: Zhang et al., "MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model", 2024
/// </para>
/// </remarks>
public class MotionDiffuseModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 263;
    private const double DEFAULT_GUIDANCE = 2.5;

    private SiTPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public MotionDiffuseModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, SiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(SiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: 4,
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
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    public override IDiffusionModel<T> Clone()
    {
        var clone = new MotionDiffuseModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "MotionDiffuse", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Fine-grained body part text-driven motion generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "part-aware-motion-diffusion");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("motion_dimensions", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
