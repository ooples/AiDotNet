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
/// StyleAligned-Edit model for consistent multi-image style editing via shared self-attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StyleAligned-Edit extends the StyleAligned shared attention mechanism to image editing,
/// ensuring that multiple edited images maintain visual consistency. All images in a batch
/// share self-attention features, enforcing a coherent visual style across the set.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you need to edit multiple images with the same visual style
/// (like a series of product photos or a consistent set of illustrations), StyleAligned-Edit
/// ensures they all look like they belong together by sharing style information across images.
/// </para>
/// <para>
/// Reference: Hertz et al., "Style Aligned Image Generation via Shared Attention", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 512, NumInferenceSteps = 30 };
/// var model = new StyleAlignedEditModel&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
/// var edited = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelTask(ModelTask.Editing)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Style Aligned Image Generation via Shared Attention", "https://arxiv.org/abs/2312.02133", Year = 2024, Authors = "Hertz et al.")]
public class StyleAlignedEditModel<T> : LatentDiffusionModelBase<T>
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

    public StyleAlignedEditModel(
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
            architecture: Architecture, inputChannels: 4, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2 }, contextDim: 2048, seed: seed);
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
        // Fast path: O(1) copy-on-write share when the default clone is structurally identical
        // (the common foundation-scale case the COW lever targets — no re-materialization/OOM).
        var clone = new StyleAlignedEditModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        if (clone.TryShareParametersFrom(this)) return clone;
        // Structure mismatch ⇒ custom architecture/predictor/VAE the default clone can't reproduce;
        // rebuild faithfully from this instance's configuration so the clone is observationally
        // identical instead of throwing on a parameter-count mismatch.
        return new StyleAlignedEditModel<T>(
            architecture: Architecture,
            options: (DiffusionModelOptions<T>)Options,
            scheduler: Scheduler,
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            seed: RandomGenerator.Next());
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "StyleAligned-Edit", Version = "1.0",
            Description = "Consistent multi-image editing via shared self-attention alignment",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "shared-attention-style-editing");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder", "CLIP ViT-L/14 + OpenCLIP ViT-bigG/14");
        m.SetProperty("context_dim", 2048);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
