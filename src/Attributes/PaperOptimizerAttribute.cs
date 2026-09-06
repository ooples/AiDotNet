using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares the training recipe a model's research paper specifies — the optimizer, its
/// hyperparameters, the learning-rate schedule and gradient clipping — so that a model given no
/// optimizer trains the way its paper says rather than at an arbitrary library default.
/// </summary>
/// <remarks>
/// <para>
/// Sits alongside <see cref="ResearchPaperAttribute"/> in the declarative block a model already
/// carries, so the citation and the recipe it produced are read together and can be checked
/// against each other:
/// </para>
/// <code>
/// [ResearchPaper("Deep Residual Learning for Image Recognition",
///                "https://arxiv.org/abs/1512.03385", Year = 2016, Authors = "He et al.")]
/// [PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 0.1, Momentum = 0.9,
///                 WeightDecay = 1e-4, Schedule = LearningRateSchedulerType.ReduceOnPlateau,
///                 Source = "Sec. 3.4 (Implementation)")]
/// public class ResNetNetwork&lt;T&gt; : ...
/// </code>
/// <para><b>For Beginners:</b> A paper does not just give a learning rate — it gives a whole
/// recipe: which optimizer, what learning rate, how that rate changes over training, and how
/// gradients are clipped. Those parts only work together. ResNet's paper uses SGD at 0.1; feeding
/// 0.1 to Adam instead would diverge immediately. So the recipe is declared as a unit, and the
/// library builds the optimizer the paper actually used.
/// </para>
/// <para>
/// <b>The whole recipe, not just the rate.</b> An earlier revision recorded only scalars and
/// applied them to whatever optimizer the model happened to construct, skipping the declaration
/// when the kinds disagreed. That was the wrong shape: it left a model with the wrong optimizer
/// and the wrong schedule while appearing to be paper-faithful. Resolution now CONSTRUCTS the
/// declared optimizer.
/// </para>
/// <para>
/// <b>Only declare what the paper states.</b> Every hyperparameter is an optional named property,
/// and anything left unset falls back to the library default. An unset property reads as "the
/// paper does not say"; a set one is a claim about the literature. <see cref="Source"/> exists to
/// make that discipline checkable — a value that cannot be pointed at a section should not be
/// declared, because an invented number looks authoritative and nobody re-checks it.
/// </para>
/// <para>
/// <b>Precedence.</b> A caller's <c>ConfigureOptimizer</c> always wins. This supplies the default
/// for the case where nothing was configured at all.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class PaperOptimizerAttribute : Attribute
{
    /// <summary>Declares the training recipe this model's paper specifies.</summary>
    /// <param name="optimizer">The optimizer the paper trains with. This one is built.</param>
    public PaperOptimizerAttribute(OptimizerKind optimizer)
    {
        Optimizer = optimizer;
    }

    /// <summary>The optimizer the paper specifies, and the one that will be constructed.</summary>
    public OptimizerKind Optimizer { get; }

    /// <summary>
    /// Where in the paper this recipe comes from, for example <c>"Sec. 4.1, Table 8"</c>.
    /// </summary>
    /// <remarks>
    /// Required whenever anything is declared, and enforced by AIDN102. It is the anti-fabrication
    /// guard, and it lets a reviewer verify an entry without re-deriving it — the difference
    /// between a claim that can be audited and one that merely looks confident.
    /// </remarks>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Which model size or configuration variant this recipe applies to, matched against
    /// <see cref="AiDotNet.Interfaces.IPaperOptimizerVariant.PaperOptimizerVariant"/>.
    /// </summary>
    /// <remarks>
    /// Papers routinely give different settings per size, so one declaration per class cannot be
    /// faithful. Repeat the attribute keyed by variant, preferring <c>nameof</c> over a literal so
    /// a renamed enum member is a compile error rather than a silently unmatched key. An attribute
    /// with no variant is the fallback for every variant lacking its own entry, so partial
    /// population is safe.
    /// </remarks>
    public string Variant { get; set; } = string.Empty;

    /// <summary>
    /// Which part of a composite model this recipe applies to, for example
    /// <c>"discriminator"</c>. Empty means the model as a whole.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Papers for composite models state different settings per part. Stable Audio Open gives a
    /// base learning rate of 1.5e-4 for its autoencoder, 3e-4 for its discriminators and 5e-5 for
    /// its DiT; a GAN paper routinely separates generator from discriminator. One recipe per model
    /// cannot express any of that, and picking whichever number appears first would be a silent
    /// mis-declaration of the rest.
    /// </para>
    /// <para>
    /// The call site names the component it is building --
    /// <c>PaperOptimizerFactory.CreateFor(this, "discriminator")</c> -- and matching is exact and
    /// case-insensitive. A component with no declaration of its own falls back to the unnamed
    /// recipe, so a model can declare a shared default and override only the parts that differ.
    /// </para>
    /// </remarks>
    public string Component { get; set; } = string.Empty;

    // ---- Optimizer hyperparameters -------------------------------------------------------

    /// <summary>The paper's learning rate. Unset means the paper does not state a constant one.</summary>
    /// <remarks>
    /// Leave unset when the rate is produced by a formula rather than a constant — the Transformer's
    /// warmup schedule, for instance. Declare <see cref="Schedule"/> instead.
    /// </remarks>
    public double LearningRate { get; set; } = double.NaN;

    /// <summary>
    /// The paper's weight decay. Unset means unstated.
    /// </summary>
    /// <remarks>
    /// Worth declaring explicitly as <c>0</c> when a paper specifies plain Adam, because AdamW
    /// otherwise contributes its own decoupled decay of 0.01 to every parameter on every step —
    /// the defect commit <c>1972a510a</c> fixed in <c>SpanBasedNERBase</c>. Note also that L2
    /// regularization in Adam and decoupled decay in AdamW are not the same operation, which is
    /// another reason the optimizer must be declared alongside the number.
    /// </remarks>
    public double WeightDecay { get; set; } = double.NaN;

    /// <summary>First moment decay (Adam-family beta1). Unset means unstated.</summary>
    public double Beta1 { get; set; } = double.NaN;

    /// <summary>Second moment decay (Adam-family beta2). Unset means unstated.</summary>
    public double Beta2 { get; set; } = double.NaN;

    /// <summary>Numerical-stability epsilon. Unset means unstated.</summary>
    public double Epsilon { get; set; } = double.NaN;

    /// <summary>Momentum coefficient, for SGD-momentum, RMSProp and friends. Unset means unstated.</summary>
    public double Momentum { get; set; } = double.NaN;

    /// <summary>Whether the paper uses Nesterov momentum.</summary>
    public bool UseNesterov { get; set; }

    /// <summary>RMSProp's decay / smoothing constant (often written rho or alpha). Unset means unstated.</summary>
    public double Rho { get; set; } = double.NaN;

    // ---- Learning-rate schedule ----------------------------------------------------------

    /// <summary>
    /// The learning-rate schedule the paper uses.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The schedule is part of the recipe, not an implementation detail. A post-LN transformer
    /// trained without warmup diverges at the same learning rate that works with it, and a CNN
    /// trained at a constant rate lands materially worse than one with step decay. Declaring the
    /// rate while ignoring the schedule reproduces neither.
    /// </para>
    /// <para>
    /// Left at <see cref="LearningRateSchedulerType.Constant"/> means the paper trains at a fixed
    /// rate, or does not say. The schedule's parameters are the properties below; which ones apply
    /// depends on the type chosen.
    /// </para>
    /// </remarks>
    public LearningRateSchedulerType Schedule { get; set; } = LearningRateSchedulerType.Constant;

    /// <summary>Warmup steps before the main schedule begins. Unset means no warmup.</summary>
    /// <remarks>
    /// The Transformer's 4000-step warmup is the canonical example; it is the difference between
    /// training and diverging, not a tuning nicety.
    /// </remarks>
    public int WarmupSteps { get; set; }

    /// <summary>
    /// What the rate does AFTER warmup finishes, when <see cref="Schedule"/> is
    /// <see cref="LearningRateSchedulerType.LinearWarmup"/>.
    /// </summary>
    /// <remarks>
    /// Warmup and the decay that follows it are two halves of one published curve, and the second
    /// half is the half that is usually dropped. Whisper warms up over 2048 updates and then decays
    /// linearly to zero (Radford et al. 2022, Table 17); holding the rate flat after warmup would
    /// reproduce the first 0.2% of that schedule and none of the rest. Left at
    /// <c>Constant</c> the rate simply holds, which is right for papers that only specify warmup.
    /// </remarks>
    public LinearWarmupScheduler.DecayMode PostWarmupDecay { get; set; }
        = LinearWarmupScheduler.DecayMode.Constant;

    /// <summary>
    /// Warmup expressed as a fraction of the whole run, for papers that state it that way.
    /// Unset means the paper gives an absolute step count, or no warmup.
    /// </summary>
    /// <remarks>
    /// HuBERT ramps up over the first 8% of training steps (Hsu et al. 2021, Sec. IV-A), which no
    /// absolute number can represent: 8% of a 400k-step pre-training run and 8% of a short
    /// fine-tune are different counts and both are what the paper says. Stated as a fraction it
    /// stays exact at any run length, where a transcribed step count would be wrong at every length
    /// but one. Takes precedence over <see cref="WarmupSteps"/> when both are declared.
    /// </remarks>
    public double WarmupFraction { get; set; } = double.NaN;

    /// <summary>
    /// How much of the run is spent holding the peak rate before decay begins, for
    /// <see cref="LearningRateSchedulerType.TriStage"/>. Unset means no hold phase.
    /// </summary>
    /// <remarks>
    /// wav2vec 2.0 fine-tunes with warmup over the first 10% of updates, a constant hold for the
    /// next 40%, and linear decay for the remainder (Baevski et al. 2020, Sec. 4.3). The hold is
    /// not a detail: without it the rate begins falling four times earlier than published.
    /// </remarks>
    public double HoldFraction { get; set; } = double.NaN;

    /// <summary>
    /// Which cyclic policy the paper uses, for <see cref="LearningRateSchedulerType.Cyclic"/>.
    /// </summary>
    /// <remarks>
    /// The policies differ in amplitude: triangular2 halves the range on every cycle where
    /// triangular keeps it, so after four cycles -- what ECAPA-TDNN trains for (Desplanques et al.
    /// 2020, Sec. 3) -- the two have drifted apart by 8x. For a cyclic schedule
    /// <see cref="LearningRate"/> is the upper bound and <see cref="MinLearningRate"/> the lower,
    /// with <see cref="StepSize"/> the half-cycle, which is how these papers state them.
    /// </remarks>
    public CyclicLRScheduler.CyclicMode CyclicPolicy { get; set; }
        = CyclicLRScheduler.CyclicMode.Triangular;

    /// <summary>Multiplicative decay factor, for exponential and step schedules. Unset means unstated.</summary>
    public double DecayRate { get; set; } = double.NaN;

    /// <summary>Interval, in steps or epochs, between decay events. Unset means unstated.</summary>
    public int StepSize { get; set; }

    /// <summary>
    /// The exact steps a paper decays at, for schedules stated as a list rather than an interval.
    /// Empty means the paper gives an interval, or no step decay.
    /// </summary>
    /// <remarks>
    /// Segment Anything decreases the rate by 10x at 60,000 and again at 86,666 iterations (Kirillov
    /// et al. 2023, Training recipe) -- points that are not evenly spaced, so no
    /// <see cref="StepSize"/> interval can describe them.
    /// </remarks>
    public int[] Milestones { get; set; } = [];

    /// <summary>Floor the schedule decays towards. Unset means unstated.</summary>
    public double MinLearningRate { get; set; } = double.NaN;

    /// <summary>
    /// The batch size the paper's learning rate was chosen for. Unset means unstated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A learning rate is only meaningful alongside the batch it was tuned for. MobileNetV3's 0.1
    /// is stated for batch 4096; applied at batch 32 it is roughly two orders of magnitude too
    /// large, and training does not converge. Declaring the reference batch lets the library apply
    /// the linear scaling rule -- multiply the rate by the ratio of actual to reference batch
    /// (Goyal et al. 2017, "Accurate, Large Minibatch SGD") -- instead of transplanting a number
    /// into a regime it was never chosen for.
    /// </para>
    /// <para>
    /// Left unset, the declared rate is used as-is. That is the right default for papers that
    /// state a rate without tying it to a large batch.
    /// </para>
    /// </remarks>
    public int ReferenceBatchSize { get; set; }

    // ---- Gradient clipping ---------------------------------------------------------------

    /// <summary>The paper's gradient-norm clip. Unset means the paper does not clip.</summary>
    /// <remarks>
    /// Frequently specified for transformers and recurrent models, and frequently the reason a
    /// reproduction is stable or not.
    /// </remarks>
    public double MaxGradientNorm { get; set; } = double.NaN;

    /// <summary>True when this declaration states anything beyond the optimizer's identity.</summary>
    /// <remarks>
    /// <c>NaN</c> is the "unset" marker because it is the one double that cannot be a legitimate
    /// hyperparameter, so no real paper value is mistaken for an omission. Comparing against 0
    /// would misread a deliberately declared <c>WeightDecay = 0</c> — precisely the case that
    /// matters most.
    /// </remarks>
    public bool DeclaresAnyHyperparameter
        => !double.IsNaN(LearningRate)
        || !double.IsNaN(WeightDecay)
        || !double.IsNaN(Beta1)
        || !double.IsNaN(Beta2)
        || !double.IsNaN(Epsilon)
        || !double.IsNaN(Momentum)
        || !double.IsNaN(Rho)
        || !double.IsNaN(MinLearningRate)
        || !double.IsNaN(MaxGradientNorm)
        || UseNesterov
        || WarmupSteps > 0
        || !double.IsNaN(WarmupFraction)
        || !double.IsNaN(HoldFraction)
        || StepSize > 0
        || Milestones.Length > 0
        || CyclicPolicy != CyclicLRScheduler.CyclicMode.Triangular
        || PostWarmupDecay != LinearWarmupScheduler.DecayMode.Constant
        || !double.IsNaN(DecayRate)
        || Schedule != LearningRateSchedulerType.Constant;
}
