using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares the optimizer and hyperparameters a model's research paper specifies, so the model
/// trains at its paper's settings instead of the optimizer class's generic defaults.
/// </summary>
/// <remarks>
/// <para>
/// Sits alongside <see cref="ResearchPaperAttribute"/> in the declarative block a model already
/// carries, so the citation and the numbers it produced are read together and can be checked
/// against each other:
/// </para>
/// <code>
/// [ResearchPaper("InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions",
///                "https://arxiv.org/abs/2211.05778", Year = 2023, Authors = "Wang et al.")]
/// [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.05,
///                 Source = "Sec. 4.1, Table 8")]
/// public partial class InternImage&lt;T&gt; : ...
/// </code>
/// <para><b>For Beginners:</b> Every model in this library is an implementation of a published
/// paper, and papers specify how to train the model — which optimizer, what learning rate, how
/// much weight decay. Without this attribute a model silently trains at whatever the optimizer
/// class happens to default to, which is rarely what the paper used, so results do not match the
/// published ones. This attribute records the paper's answer.
/// </para>
/// <para>
/// <b>Only declare what the paper actually states.</b> Every hyperparameter here is an optional
/// named property, and anything left unset falls back to the library default. An unset property
/// reads as "the paper does not say"; a set one is a claim about the literature. Declaring a value
/// the paper never gave is worse than leaving it out, because an invented number looks
/// authoritative and nobody re-checks it. <see cref="Source"/> exists to make that discipline
/// checkable.
/// </para>
/// <para>
/// <b>Precedence.</b> A caller's <c>ConfigureOptimizer</c> options always win. This attribute only
/// supplies defaults for the case where a model constructs its optimizer with no options at all —
/// the situation issue #1928 describes, where 685 construction sites silently inherited the
/// optimizer class's own defaults.
/// </para>
/// <para>
/// <b>Scope.</b> Scalar hyperparameters only. Schedule shape (warm-up, cosine decay) is not
/// expressible as attribute arguments and stays in code; warm-up-and-decay is already the library
/// default rather than a flat rate.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class PaperOptimizerAttribute : Attribute
{
    /// <summary>
    /// Declares the optimizer this model's paper trains with.
    /// </summary>
    /// <param name="optimizer">The optimizer the paper specifies.</param>
    public PaperOptimizerAttribute(OptimizerKind optimizer)
    {
        Optimizer = optimizer;
    }

    /// <summary>The optimizer the paper specifies.</summary>
    public OptimizerKind Optimizer { get; }

    /// <summary>
    /// Where in the paper these values come from, for example <c>"Sec. 4.1, Table 8"</c>.
    /// </summary>
    /// <remarks>
    /// Required whenever any hyperparameter is declared. It is the anti-fabrication guard: a value
    /// that cannot be pointed at a section of the paper should not be declared at all. It also lets
    /// a reviewer verify an entry without re-deriving it, which is the difference between a claim
    /// that can be audited and one that merely looks confident.
    /// </remarks>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Which model size or configuration variant these values apply to, matched against
    /// <see cref="AiDotNet.Interfaces.IPaperOptimizerVariant.PaperOptimizerVariant"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Papers routinely give different settings per size — InternImage-T and InternImage-H do not
    /// share a learning rate — so one declaration per class cannot be faithful. Repeat the
    /// attribute, keyed by variant, and prefer <c>nameof</c> over a literal so a renamed enum member
    /// is a compile error rather than a silently unmatched key:
    /// </para>
    /// <code>
    /// [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4,
    ///                 Variant = nameof(InternImageModelSize.Tiny), Source = "Table 8")]
    /// [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5,
    ///                 Variant = nameof(InternImageModelSize.Huge), Source = "Table 8")]
    /// </code>
    /// <para>
    /// An attribute with no <see cref="Variant"/> is the fallback for every variant that has no
    /// entry of its own, so partial population is expected and safe: declared variants use their
    /// own values, the rest use the unkeyed entry, and models with neither keep library defaults.
    /// </para>
    /// </remarks>
    public string Variant { get; set; } = string.Empty;

    /// <summary>The paper's learning rate. Unset means the paper does not state one.</summary>
    public double LearningRate { get; set; } = double.NaN;

    /// <summary>
    /// The paper's weight decay. Unset means the paper does not state one.
    /// </summary>
    /// <remarks>
    /// Worth declaring explicitly as <c>0</c> when a paper specifies plain Adam, because AdamW
    /// otherwise contributes its own decoupled decay of 0.01 to every parameter on every step. That
    /// is not hypothetical: it is exactly the defect commit <c>1972a510a</c> fixed in
    /// <c>SpanBasedNERBase</c>, where no span-NER paper asks for that decay.
    /// </remarks>
    public double WeightDecay { get; set; } = double.NaN;

    /// <summary>The paper's first moment decay (Adam-family beta1). Unset means unstated.</summary>
    public double Beta1 { get; set; } = double.NaN;

    /// <summary>The paper's second moment decay (Adam-family beta2). Unset means unstated.</summary>
    public double Beta2 { get; set; } = double.NaN;

    /// <summary>The paper's numerical-stability epsilon. Unset means unstated.</summary>
    public double Epsilon { get; set; } = double.NaN;

    /// <summary>True when this declaration states at least one hyperparameter.</summary>
    /// <remarks>
    /// <c>NaN</c> is the "unset" marker because it is the one double value that cannot be a
    /// legitimate hyperparameter, so no real paper value is mistaken for an omission. Comparing
    /// against 0 would misread a deliberately declared <c>WeightDecay = 0</c> as unstated, which is
    /// precisely the case that matters most.
    /// </remarks>
    public bool DeclaresAnyHyperparameter
        => !double.IsNaN(LearningRate)
        || !double.IsNaN(WeightDecay)
        || !double.IsNaN(Beta1)
        || !double.IsNaN(Beta2)
        || !double.IsNaN(Epsilon);
}
