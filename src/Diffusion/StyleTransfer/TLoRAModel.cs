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

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// T-LoRA: timestep-dependent low-rank adaptation for customizing a diffusion model from a SINGLE
/// image without overfitting to it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Soboleva, Alanov, Kuznetsov and Sobolev, "T-LoRA: Single Image Diffusion Model Customization
/// Without Overfitting" (arXiv:2507.05964). Fine-tuning "frequently suffers from overfitting when
/// training samples are limited, compromising both generalization capability and output diversity",
/// and this paper takes the hardest version of that: adapting "using just a single concept image".
/// </para>
/// <para>
/// <b>PURPOSE CHANGE.</b> This class previously described "temporal LoRA-based style transfer with
/// video-consistent stylization" — applying an artistic style across video frames without flicker.
/// The real T-LoRA is not that. It is single-image CONCEPT personalization: teaching a model a new
/// object from one photo while keeping its diversity. The two share only the letters. The citation
/// this replaced pointed at arXiv:2405.12345, which is a functional-analysis paper on the learning
/// behaviour of paradise fish.
/// </para>
/// <para>
/// <b>Two innovations, both implemented in <see cref="TimestepDependentLora{T}"/>.</b>
/// </para>
/// <list type="number">
/// <item><description><b>A timestep-dependent rank schedule.</b> "Higher diffusion timesteps are
/// more prone to overfitting than lower ones", so the adapter is given fewer directions exactly
/// there and full rank at low timesteps where it is refining detail.</description></item>
/// <item><description><b>Orthogonal initialization</b>, "a weight parametrization technique that
/// ensures independence between adapter components". This is what makes the schedule bite: masking
/// the tail of a set of CORRELATED directions removes no capacity, because the survivors still span
/// what was masked.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Show a picture generator one photo of your dog and ask it to learn "your
/// dog", and it will usually memorize that exact photo — same pose, same background, every time. The
/// memorizing happens mainly in the noisy early stage of generation, so this gives the model fewer
/// adjustable directions during that stage and all of them later, when it is only adding detail.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
// Deliberately NOT ModelTask.StyleTransfer, which the pre-rebuild version claimed. T-LoRA is
// single-image CONCEPT personalization — learning a new subject from one photo — not style transfer.
// The class stays in the StyleTransfer folder only because moving it is a separate, wider change.
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("T-LoRA: Single Image Diffusion Model Customization Without Overfitting",
    "https://arxiv.org/abs/2507.05964",
    Year = 2025,
    Authors = "Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev")]
public class TLoRAModel<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// The latent channel count used when the caller does not specify one: the Stable-Diffusion
    /// value, which the default predictor and VAE below are both built around.
    /// </summary>
    /// <remarks>
    /// A DEFAULT, not a constant of the model. Callers override it via
    /// <see cref="DiffusionModelOptions{T}.LatentChannels"/>; see <see cref="_latentChannels"/>.
    /// </remarks>
    private const int DEFAULT_LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;

    private readonly int _latentChannels;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => _latentChannels;
    public override long ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public TLoRAModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        int requested = options?.LatentChannels ?? DEFAULT_LATENT_CHANNELS;
        if (requested <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), requested,
                "DiffusionModelOptions.LatentChannels must be positive when specified.");
        }

        // An injected VAE has already fixed the latent width, so an override that disagrees with it
        // cannot be honoured. Say so here rather than letting it surface as a shape mismatch deep
        // inside the denoising loop, where the connection back to this option is not obvious.
        if (vae is not null && options?.LatentChannels is int overridden && vae.LatentChannels != overridden)
        {
            throw new ArgumentException(
                $"DiffusionModelOptions.LatentChannels was set to {overridden}, but the supplied VAE " +
                $"produces {vae.LatentChannels} latent channels. Either omit the override or supply a " +
                "VAE built for the same width.", nameof(options));
        }

        // A supplied VAE wins when no explicit override was given: it is the authority on the width
        // the rest of the pipeline has to agree with.
        _latentChannels = vae?.LatentChannels ?? requested;
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: _latentChannels, outputChannels: _latentChannels,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: _latentChannels,
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
        // Clone the ACTUAL predictor/VAE (see InstaFlowModel/MultiDiffusionModel): passing only
        // conditioner/seed rebuilt InitializeLayers' DEFAULT-sized, lazily-unresolved sub-models, so once
        // the source resolved its lazy layers via a forward pass the trainable-layer shapes no longer
        // lined up 1:1 and Clone diverged. Cloning the resolved predictor/VAE (+ same architecture/
        // options/scheduler) makes the clone structurally identical.
        var clone = new TLoRAModel<T>(
            architecture: Architecture,
            options: Options as DiffusionModelOptions<T>,
            scheduler: Scheduler,
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            seed: null);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameterChunks(GetParameterChunks());
        return clone;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "T-LoRA", Version = "1.0",
            // Was "Temporal LoRA-based style transfer with video consistency" — the fabricated purpose
            // this class carried before the rebuild. T-LoRA is single-image concept customization.
            Description = "Timestep-dependent low-rank adaptation for single-image diffusion customization",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "timestep-dependent-lora");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", _latentChannels);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
