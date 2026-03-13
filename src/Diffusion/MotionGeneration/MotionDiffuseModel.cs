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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model", "https://arxiv.org/abs/2208.15001", Year = 2024, Authors = "Zhang et al.")]
public class MotionDiffuseModel<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// Number of motion feature dimensions per frame (263 = 3 root velocity + 6*N joint rotations + ...).
    /// This is the motion representation size, not the VAE latent channel count.
    /// </summary>
    private const int MOTION_FEATURE_DIM = 263;
    private const int VAE_LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 2.5;

    private SiTPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => VAE_LATENT_CHANNELS;
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
        m.SetProperty("motion_dimensions", MOTION_FEATURE_DIM);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
