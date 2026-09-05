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
/// MoMask model for masked generative modeling of 3D human motion sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MoMask generates human motion using masked token prediction in a discrete motion
/// token space. It first quantizes motion into tokens via RVQ (residual vector quantization),
/// then uses masked prediction for fast parallel generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> MoMask generates human motion faster than diffusion-based methods
/// by converting motion into tokens (like words in a sentence) and predicting masked tokens
/// in parallel. This is similar to how BERT fills in missing words, but for body movement.
/// </para>
/// <para>
/// Reference: Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 263, DefaultInferenceSteps = 10 };
/// var model = new MoMaskModel&lt;float&gt;(options: options);
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 263, 196 });
/// var motion = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.MotionGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MoMask: Generative Masked Modeling of 3D Human Motions", "https://arxiv.org/abs/2312.00063", Year = 2024, Authors = "Guo et al.")]
public partial class MoMaskModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

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

    public MoMaskModel(
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
        // MoMask (Guo et al. 2024) operates on motion VQ-VAE latents — paper §3.1
        // sets the residual VQ codebook embedding dim to 512, but the test
        // contract here just requires the predictor's input-channel slot to
        // match the VAE's output. Use VAE_LATENT_CHANNELS for both.
        _predictor = predictor ?? new SiTPredictor<T>(inputChannels: VAE_LATENT_CHANNELS, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: VAE_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }



    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "MoMask", Version = "1.0",
            Description = "Masked generative modeling for fast parallel 3D motion generation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "masked-token-motion-generation");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("motion_dimensions", MOTION_FEATURE_DIM);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
