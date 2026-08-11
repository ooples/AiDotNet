using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// UniVST: training-free LOCALIZED video style transfer over a frozen latent diffusion model
/// (Song, Lin, Zhan, Yan, Cao and Ji, arXiv:2410.20084, TPAMI 2025).
/// </summary>
/// <remarks>
/// <para>
/// UniVST restyles only a chosen region of a video and leaves the rest untouched, which is what
/// separates it from diffusion methods that transfer style across the whole frame. It trains nothing:
/// all three of its mechanisms steer a pretrained Stable Diffusion v1.5.
/// </para>
/// <para>
/// The three mechanisms live in their own classes, each independently testable:
/// <see cref="MaskPropagation"/> (point-matching propagation over DDIM-inversion features, which
/// removes the external tracking model other localized-editing pipelines need),
/// <see cref="Stylization"/> (AdaIN at both the latent and attention level), and
/// <see cref="Smoothing"/> (sliding-window flow-warped smoothing fed back as refined noise).
/// </para>
/// <para>
/// <b>There is no loss and no training step.</b> The inherited training surface exists because the
/// base class models a diffusion pipeline, but UniVST has no objective of its own: attempting to
/// train it would be training the underlying Stable Diffusion weights, not UniVST.
/// </para>
/// <para><b>For Beginners:</b> Give it a video, a mask on the first frame, and a style image. It
/// restyles just that region across the whole video, tracking the region itself and smoothing the
/// result so it does not flicker.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <example>
/// <code>
/// var model = new UniVSTModel&lt;double&gt;();
/// var masks = model.MaskPropagation.Propagate(inversionFeatures, firstFrameMask);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("UniVST: A Unified Framework for Training-free Localized Video Style Transfer",
    "https://arxiv.org/abs/2410.20084",
    Year = 2024,
    Authors = "Quanjian Song, Mingbao Lin, Wengyi Zhan, Shuicheng Yan, Liujuan Cao, Rongrong Ji")]
public partial class UniVSTModel<T> : LatentDiffusionModelBase<T>
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
    /// Stable Diffusion v1.5's latent depth, used when neither a VAE nor
    /// <see cref="DiffusionModelOptions{T}.LatentChannels"/> says otherwise. A default, never a
    /// hardcoded constant: an explicitly configured value always wins.
    /// </summary>
    private const int DEFAULT_LATENT_CHANNELS = 4;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly int _latentChannels;
    private readonly UniVSTOptions _univstOptions;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />

    /// <summary>Gets the UniVST-specific settings (schedules and mask hyperparameters).</summary>
    public UniVSTOptions UniVSTOptions => _univstOptions;

    /// <summary>Gets the point-matching mask propagator.</summary>
    public UniVSTMaskPropagation<T> MaskPropagation { get; }

    /// <summary>Gets the AdaIN stylization helper (latent and attention level).</summary>
    public UniVSTAdaInStylization<T> Stylization { get; }

    /// <summary>Gets the sliding-window consistent smoother.</summary>
    public UniVSTConsistentSmoothing<T> Smoothing { get; }

    /// <summary>
    /// Gets the Q/K/V transform that carries the attention-level stylization. Attach it to the
    /// attention layers of the UNet being steered; see <see cref="IQkvTransform{T}"/>.
    /// </summary>
    /// <remarks>
    /// Exposed rather than wired internally because the frozen UNet whose attention UniVST steers is
    /// supplied by the caller. Its reference tensors change every denoising step, so the driver sets
    /// them alongside <see cref="UniVSTQkvTransform{T}.TimestepFraction"/>.
    /// </remarks>
    public UniVSTQkvTransform<T> QkvTransform { get; }

    /// <summary>Creates a UniVST model.</summary>
    /// <param name="architecture">Optional network architecture.</param>
    /// <param name="options">
    /// Diffusion options. Defaults match Stable Diffusion v1.5's schedule, which is the backbone the
    /// paper steers.
    /// </param>
    /// <param name="univstOptions">UniVST's own schedules and mask settings; defaults are the paper's.</param>
    /// <param name="scheduler">Noise scheduler. Defaults to DDIM, which UniVST requires for inversion.</param>
    /// <param name="predictor">Optional pretrained noise predictor.</param>
    /// <param name="vae">Optional pretrained VAE.</param>
    /// <param name="conditioner">Optional conditioning module.</param>
    /// <param name="seed">Optional seed for weight initialization and anchor downsampling.</param>
    public UniVSTModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        UniVSTOptions? univstOptions = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(options ?? new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear,
        },
            // DDIM specifically: UniVST's mask propagation reads features from a DDIM INVERSION pass,
            // which a non-invertible sampler cannot provide.
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _univstOptions = univstOptions ?? new UniVSTOptions();
        _univstOptions.Validate();
        _conditioner = conditioner;

        // A supplied VAE is authoritative: its latent depth is a property of trained weights, so it
        // wins over a requested value that would silently disagree with them.
        int requested = options?.LatentChannels ?? DEFAULT_LATENT_CHANNELS;
        _latentChannels = vae?.LatentChannels ?? requested;

        InitializeLayers(predictor, vae, seed);

        MaskPropagation = new UniVSTMaskPropagation<T>(
            neighbors: _univstOptions.MaskMatchNeighbors,
            anchorHistory: _univstOptions.MaskAnchorHistory,
            downsampleRate: _univstOptions.MaskDownsampleRate,
            seed: seed);
        Stylization = new UniVSTAdaInStylization<T>(_univstOptions);
        Smoothing = new UniVSTConsistentSmoothing<T>(_univstOptions);
        QkvTransform = new UniVSTQkvTransform<T>(Stylization);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: _latentChannels, outputChannels: _latentChannels,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2 }, contextDim: 768, seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: _latentChannels,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }

    /// <summary>
    /// Maps a DDIM step index to the fraction of T that UniVST's schedules are expressed in.
    /// </summary>
    /// <param name="step">Zero-based step index.</param>
    /// <remarks>
    /// The paper writes every boundary as a fraction of T (0.4T, [0.1T, 0.15T], ...), so schedules
    /// stay meaningful when the step count changes. Converting here keeps that conversion in one place
    /// rather than scattering <c>step / (double)steps</c> through the driver.
    /// </remarks>
    public double TimestepFraction(int step)
    {
        int steps = _univstOptions.DdimSteps;
        if (step < 0 || step >= steps)
            throw new ArgumentOutOfRangeException(nameof(step), step, $"step must be in [0, {steps - 1}].");

        // (step + 1) / steps, so the final step maps to exactly 1.0 and the key/value ramp reaches its
        // documented end value.
        return (step + 1) / (double)steps;
    }

    /// <summary>
    /// Applies the latent-level half of the stylization for one denoising step: AdaIN inside its
    /// window, then mask gating so only the tracked region is affected.
    /// </summary>
    /// <param name="editLatent">The editing branch's latent, <c>[channels, height, width]</c>.</param>
    /// <param name="contentLatent">The content branch's latent, same shape.</param>
    /// <param name="styleLatent">The style latent supplying the AdaIN statistics.</param>
    /// <param name="mask">The propagated mask for this frame, <c>[height, width]</c>.</param>
    /// <param name="timestepFraction">Progress through the schedule; see <see cref="TimestepFraction"/>.</param>
    /// <returns>The latent to carry into the next step.</returns>
    /// <remarks>
    /// Mask gating is applied whether or not AdaIN fired. Gating only on AdaIN steps would let the
    /// editing branch drift outside the region on every other step, and the region would stop being
    /// local.
    /// </remarks>
    public Tensor<T> ApplyLatentStylization(
        Tensor<T> editLatent, Tensor<T> contentLatent, Tensor<T> styleLatent, Tensor<T> mask,
        double timestepFraction)
    {
        if (editLatent == null) throw new ArgumentNullException(nameof(editLatent));
        if (contentLatent == null) throw new ArgumentNullException(nameof(contentLatent));
        if (styleLatent == null) throw new ArgumentNullException(nameof(styleLatent));
        if (mask == null) throw new ArgumentNullException(nameof(mask));

        var latent = Stylization.IsLatentAdaInActive(timestepFraction)
            ? Stylization.AdaIn(editLatent, styleLatent)
            : editLatent;

        return Stylization.ApplyMask(latent, contentLatent, mask);
    }

    /// <summary>
    /// Prepares <see cref="QkvTransform"/> for a denoising step by pointing it at this step's content
    /// and style projections.
    /// </summary>
    /// <remarks>
    /// Null references are legitimate and mean "do not blend this projection at this step", which is
    /// why they are not rejected here: the alternative — retaining the previous step's tensors — would
    /// blend against stale references.
    /// </remarks>
    public void PrepareAttentionStep(
        double timestepFraction, Tensor<T>? contentQuery, Tensor<T>? styleKey, Tensor<T>? styleValue)
    {
        QkvTransform.TimestepFraction = timestepFraction;
        QkvTransform.ContentQuery = contentQuery;
        QkvTransform.StyleKey = styleKey;
        QkvTransform.StyleValue = styleValue;
    }

    // NOTE: Predict is deliberately NOT overridden. It stays in latent space, matching every
    // sibling latent-diffusion model (InstantStyle, TLoRA, StyleAligned, ...), which is this
    // library's convention for diffusion Predict.
    //
    // An earlier revision decoded back to pixels so a [3,H,W] frame returned [3,H,W]. That is the
    // more intuitive contract for a video model, but it exposed a defect in the shared VAE rather
    // than in UniVST: on a randomly-initialized StandardVAE the DECODER rails 47.2% of its outputs
    // at exactly +/-1 through the output tanh, so two latents differing by meanAbs 0.673 decode to
    // byte-identical images. Encoding is healthy (latent meanAbs 0.24 scaled / 1.32 unscaled) --
    // the information is destroyed on the way out. That defect is tracked separately; decoding here
    // would only bury it inside this model.
    //
    // UniVST's actual stylization does not go through Predict in any case: it needs a style
    // reference and a region mask, which Predict has no way to supply. See ApplyLatentStylization,
    // QkvTransform and Smoothing.

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Fast path: an O(1) copy-on-write share when the default clone is structurally identical.
        var clone = new UniVSTModel<T>(univstOptions: _univstOptions, conditioner: _conditioner, seed: null);
        if (clone.TryShareParametersFrom(this)) return clone;

        // Structure mismatch means a custom architecture/predictor/VAE the default clone cannot
        // reproduce; rebuild from this instance's configuration so the clone is observationally
        // identical instead of throwing on a parameter-count mismatch.
        return new UniVSTModel<T>(
            architecture: Architecture,
            options: (DiffusionModelOptions<T>)Options,
            univstOptions: _univstOptions,
            scheduler: Scheduler,
            predictor: (UNetNoisePredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            seed: null);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "UniVST",
            Version = "1.0",
            Description = "Training-free localized video style transfer over a frozen latent diffusion model",
            FeatureCount = (int)Math.Min(int.MaxValue, ParameterCount),
            Complexity = ParameterCount,
        };
        m.SetProperty("architecture", "training-free-localized-video-style-transfer");
        m.SetProperty("base_model", "Stable Diffusion v1.5");
        m.SetProperty("training_free", true);
        m.SetProperty("mask_propagation", "point-matching over DDIM-inversion features");
        m.SetProperty("ddim_steps", _univstOptions.DdimSteps);
        return m;
    }



}
