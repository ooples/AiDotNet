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
/// StyDiff model for diffusion-based artistic style transfer with content preservation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StyDiff performs style transfer by injecting style features from a reference image
/// into the diffusion process via cross-attention. Content structure is preserved through
/// DDIM inversion while the style is transferred from the reference.
/// </para>
/// <para>
/// <b>For Beginners:</b> StyDiff takes the artistic style from one image (like a painting)
/// and applies it to another image (like a photo) while keeping the photo's content intact.
/// The result looks like the original photo painted in the reference style.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 512, NumInferenceSteps = 30 };
/// var model = new StyDiffModel&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
/// var stylized = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// arXiv 2308.07863 is "StyleDiffusion"; "StyDiff" is this type's short name, not the paper's title.
[ResearchPaper("StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models", "https://arxiv.org/abs/2308.07863", Year = 2023, Authors = "Wang et al.")]
public partial class StyDiffModel<T> : LatentDiffusionModelBase<T>
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
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;

    public StyDiffModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear, Seed = 42 },
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



    public override IDiffusionModel<T> Clone()
    {
        var optionsCopy = new DiffusionModelOptions<T>((DiffusionModelOptions<T>)Options);

        // Fast path: O(1) copy-on-write share when the default clone is structurally identical
        // (the common foundation-scale case the COW lever targets — no re-materialization/OOM).
        var clone = new StyDiffModel<T>(
            architecture: Architecture,
            options: optionsCopy,
            scheduler: Scheduler,
            conditioner: _conditioner,
            seed: null);
        if (clone.TryShareParametersFrom(this)) return clone;
        // Structure mismatch ⇒ custom architecture/predictor/VAE the default clone can't reproduce;
        // rebuild faithfully from this instance's configuration so the clone is observationally
        // identical instead of throwing on a parameter-count mismatch.
        var rebuilt = new StyDiffModel<T>(
            architecture: Architecture,
            options: new DiffusionModelOptions<T>((DiffusionModelOptions<T>)Options),
            scheduler: Scheduler,
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            seed: null);
        // The cloned sub-models have the same materialized structure as the source. Bind the rebuilt
        // model to the exact same Tensor objects through DiffusionModelBase's reference-counted COW
        // path, rather than creating new CloneShared tensor wrappers. The wrappers preserve values but
        // have independent tensor identity/version state, so the CPU packed-weight cache can take a
        // different cold path in the clone; DDIM compounds that small first-step reduction difference
        // across its denoising loop (the Linux CI failure was 3.43e-5). ShareWeightsFrom preserves both
        // values and inference-cache identity while EnsureOwnWeights still detaches either model before
        // training or SetParameters, keeping clone independence.
        rebuilt.ShareWeightsFrom(this);
        return rebuilt;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "StyDiff", Version = "1.0",
            Description = "Diffusion-based style transfer with cross-attention style injection",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "ddim-inversion-style-transfer");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
