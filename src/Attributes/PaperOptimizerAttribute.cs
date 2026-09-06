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

    /// <summary>Multiplicative decay factor, for exponential and step schedules. Unset means unstated.</summary>
    public double DecayRate { get; set; } = double.NaN;

    /// <summary>Interval, in steps or epochs, between decay events. Unset means unstated.</summary>
    public int StepSize { get; set; }

    /// <summary>Floor the schedule decays towards. Unset means unstated.</summary>
    public double MinLearningRate { get; set; } = double.NaN;

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
        || StepSize > 0
        || !double.IsNaN(DecayRate)
        || Schedule != LearningRateSchedulerType.Constant;
}
