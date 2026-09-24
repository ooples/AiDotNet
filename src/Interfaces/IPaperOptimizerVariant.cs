namespace AiDotNet.Interfaces;

/// <summary>
/// Implemented by models whose paper specifies different hyperparameters per size or configuration
/// variant, so the right <see cref="AiDotNet.Attributes.PaperOptimizerAttribute"/> can be selected.
/// </summary>
/// <remarks>
/// <para>
/// A model with a single set of paper hyperparameters does not need this: one unkeyed
/// <c>[PaperOptimizer]</c> applies to every instance. Implement it only where the paper genuinely
/// differs by variant, for example InternImage-T versus InternImage-H.
/// </para>
/// <para><b>For Beginners:</b> Many models come in sizes — tiny, base, large, huge. Papers often
/// train the big ones at a smaller learning rate, because a rate that suits a small model makes a
/// large one diverge. This property tells the library which size this instance is, so it picks the
/// matching row from the paper.
/// </para>
/// <para>
/// Returning the variant as a string rather than the enum itself keeps this interface free of a
/// per-model generic parameter, and matches the attribute's <c>Variant</c>, which is written as
/// <c>nameof(SomeSizeEnum.Huge)</c> so a rename stays a compile error.
/// </para>
/// </remarks>
public interface IPaperOptimizerVariant
{
    /// <summary>
    /// The variant key for this instance, matching a <c>[PaperOptimizer(Variant = ...)]</c>
    /// declaration. Return <c>null</c> or empty to use the unkeyed declaration.
    /// </summary>
    /// <remarks>
    /// Read while the optimizer is being constructed, which for most models is inside their own
    /// constructor. Return a value derived from configuration already assigned by then — typically
    /// the size enum passed in — rather than from state produced later by training or a first
    /// forward pass, which would not yet exist.
    /// </remarks>
    string? PaperOptimizerVariant { get; }
}
