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

    /// <summary>
    /// The adapter rank used when the caller does not specify one: the paper's primary setting.
    /// </summary>
    /// <remarks>
    /// "Rank r=64 (primary experiments)", with r_min derived as 50% of it. Ranks of 4, 8, 16 and 32
    /// are also reported, so this is a default rather than a requirement; pass <c>adapterRank</c> to
    /// choose another. Narrow blocks clamp it down to their own width, since a rank above the ambient
    /// dimension cannot have independent directions.
    /// </remarks>
    private const int DEFAULT_ADAPTER_RANK = 64;

    private readonly int _latentChannels;
    private readonly int _adapterRank;
    private readonly IReadOnlyList<TLoRAAttentionAdapter<T>> _adapters;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => _latentChannels;
    /// <remarks>
    /// Includes the adapters, so this stays equal to <c>GetParameters().Length</c>. They are real
    /// trainable parameters of this model, and a count that excluded them would disagree with the
    /// vector that carries them.
    /// </remarks>
    public override long ParameterCount =>
        _predictor.ParameterCount + _vae.ParameterCount + TotalAdapterParameterCount;

    public TLoRAModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null,
        int? adapterRank = null)
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

        _adapterRank = adapterRank ?? DEFAULT_ADAPTER_RANK;
        if (_adapterRank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(adapterRank), _adapterRank,
                "The T-LoRA adapter rank must be positive.");
        }

        _adapters = InjectAdapters(seed);
    }

    /// <summary>
    /// Wraps every self- and cross-attention block in the predictor with a
    /// <see cref="TLoRAAttentionAdapter{T}"/>, which is what actually puts the paper's mechanism on
    /// the forward path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper injects into "self-attention and cross-attention layers", so both are wrapped. The
    /// adapters contribute exactly zero until trained, so injection does not change what an untrained
    /// model computes — it only gives training somewhere to go.
    /// </para>
    /// <para>
    /// The horizon handed to each adapter is the SCHEDULER's TrainTimesteps, not the options' value.
    /// The scheduler is what the denoising loop and the noise-prediction path actually consult, and
    /// the two can differ; taking the schedule from anywhere else would mask at timesteps the model
    /// never visits.
    /// </para>
    /// </remarks>
    private IReadOnlyList<TLoRAAttentionAdapter<T>> InjectAdapters(int? seed)
    {
        var adapters = new List<TLoRAAttentionAdapter<T>>();
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        int horizon = Math.Max(1, Scheduler.TrainTimesteps);

        _predictor.DecorateAttentionBlocks((block, channels, isCrossAttention) =>
        {
            // ADOPT, never re-wrap. Clone() hands the constructor an ALREADY-decorated predictor
            // (cloning the resolved sub-models is what keeps clone shapes lined up), so wrapping again
            // would stack an adapter on an adapter and double the predictor's parameter count —
            // measured as "SetParameterChunks chunk length 16448 does not match layer parameter length
            // 24704" from Clone_ShouldProduceIdenticalOutput. Injection has to be idempotent.
            if (block is TLoRAAttentionAdapter<T> alreadyAdapted)
            {
                adapters.Add(alreadyAdapted);
                return block;
            }

            // The width comes from the predictor, which recorded it when it built the block. It is
            // NOT read from GetOutputShape(): that answers differently before and after lazy shape
            // resolution, so a fresh model and its clone (built from a resolved predictor) decorated
            // DIFFERENT sets of blocks, and every parameter chunk after the first divergence landed on
            // the wrong layer — "chunk length 16448 does not match layer parameter length 24704".
            if (channels <= 0)
            {
                throw new InvalidOperationException(
                    "The predictor reported a non-positive attention width, so the T-LoRA adapter cannot " +
                    "be sized. Skipping the block instead would decorate an inconsistent subset and " +
                    "silently misalign parameter chunks between a model and its clone.");
            }

            var adapter = new TLoRAAttentionAdapter<T>(
                inner: block, channels: channels, rank: _adapterRank,
                totalTimesteps: horizon, random: random);
            adapters.Add(adapter);
            return adapter;
        });

        return adapters;
    }

    /// <summary>
    /// Points every injected adapter at <paramref name="timestep"/> before the network is walked.
    /// </summary>
    /// <remarks>
    /// The rank mask is a function of the timestep, and <c>ILayer.Forward</c> does not carry one, so
    /// the value is pushed to the adapters here. Called from <see cref="PredictNoise"/>, which is the
    /// single point every training step and every sampling step passes through.
    /// </remarks>
    private void SetAdapterTimestep(int timestep)
    {
        for (int i = 0; i < _adapters.Count; i++) _adapters[i].CurrentTimestep = timestep;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Overridden solely to publish the current timestep to the adapters. Without this the mask would
    /// be evaluated at whatever timestep happened to be set last, which for a fresh model is zero —
    /// full rank everywhere, i.e. plain LoRA, and the paper's contribution silently absent.
    /// </remarks>
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        SetAdapterTimestep(timestep);
        return base.PredictNoise(noisySample, timestep);
    }

    /// <summary>
    /// Gets the injected adapters, one per wrapped attention block.
    /// </summary>
    public IReadOnlyList<TLoRAAttentionAdapter<T>> Adapters => _adapters;

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

    /// <summary>
    /// Total adapter parameters across every injected block.
    /// </summary>
    private int TotalAdapterParameterCount
    {
        get
        {
            // Null during construction: InitializeLayers runs before injection, and ParameterCount is
            // reachable from there. Zero is the honest answer at that point — no adapters exist yet.
            if (_adapters is null) return 0;

            int total = 0;
            for (int i = 0; i < _adapters.Count; i++) total += _adapters[i].AdapterParameterCount;
            return total;
        }
    }

    /// <summary>
    /// Predictor parameters, then VAE parameters, then every adapter's state in injection order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The adapter block is appended HERE, at the model, because the adapters are deliberately
    /// transparent at the LAYER level — a decorator that changed a layer's parameter count broke the
    /// positional, length-checked pairing that every parameter-copy path in this library relies on.
    /// The model is the serialization unit, so it is the right place to own full state.
    /// </para>
    /// <para>
    /// Without this, a save/load round-trip silently discarded trained adapter weights: the reloaded
    /// model would carry the correct base network and freshly-initialized adapters, which look
    /// harmless because a fresh adapter is the identity — the model would simply have forgotten
    /// everything T-LoRA had learned, with no error anywhere.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var pp = _predictor.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(pp.Length + vp.Length + TotalAdapterParameterCount);
        for (int i = 0; i < pp.Length; i++) combined[i] = pp[i];
        for (int i = 0; i < vp.Length; i++) combined[pp.Length + i] = vp[i];

        int index = pp.Length + vp.Length;
        for (int a = 0; a < _adapters.Count; a++)
        {
            var state = _adapters[a].GetAdapterState();
            for (int i = 0; i < state.Length; i++) combined[index++] = state[i];
        }
        return combined;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int pc = checked((int)_predictor.ParameterCount);
        int vc = checked((int)_vae.ParameterCount);
        int ac = TotalAdapterParameterCount;
        long expectedTotal = (long)pc + vc + ac;
        if (parameters.Length != expectedTotal)
            throw new ArgumentException(
                $"Expected {expectedTotal} parameters ({pc} predictor + {vc} VAE + {ac} T-LoRA adapter), " +
                $"got {parameters.Length}.", nameof(parameters));
        var pp = new Vector<T>(pc);
        var vp = new Vector<T>(vc);
        for (int i = 0; i < pc; i++) pp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[pc + i];
        _predictor.SetParameters(pp);
        _vae.SetParameters(vp);

        // Adapter state, in the same injection order GetParameters wrote it.
        int index = pc + vc;
        for (int a = 0; a < _adapters.Count; a++)
        {
            int count = _adapters[a].AdapterParameterCount;
            var state = new Vector<T>(count);
            for (int i = 0; i < count; i++) state[i] = parameters[index++];
            _adapters[a].SetAdapterState(state);
        }
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
            seed: null,
            // Carry the rank across, or the clone would silently fall back to the default and end up
            // with a different adapter shape than the source — which SetParameters would then reject.
            adapterRank: _adapterRank);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameterChunks(GetParameterChunks());

        // Adapter state is deliberately NOT part of any layer's parameter vector (see
        // TLoRAAttentionAdapter.GetAdapterState), so it is copied here explicitly. Without this the
        // clone would keep its own freshly-initialized adapters: identical in shape, different in
        // value, and — because Clone passes seed: null — drawn from a different RNG, so the clone would
        // produce different output despite holding identical base weights.
        if (clone._adapters.Count != _adapters.Count)
        {
            throw new InvalidOperationException(
                $"The clone has {clone._adapters.Count} T-LoRA adapters but the source has " +
                $"{_adapters.Count}. Injection is meant to be deterministic given the same predictor " +
                "shape, so a difference here means decoration ran against a different block set.");
        }

        for (int i = 0; i < _adapters.Count; i++)
        {
            // FULL state, including the frozen initialization triplet. Copying only A/B/S leaves the
            // clone subtracting its own independently-drawn A_init/B_init/S_init, so its adapter applies
            // the difference between two unrelated initializations rather than the identity.
            clone._adapters[i].Adapter.CopyStateFrom(_adapters[i].Adapter);
        }

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
