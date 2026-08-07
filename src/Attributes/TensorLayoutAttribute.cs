using System;
using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares one tensor layout a layer or model accepts as input, or produces as output, as an ordered
/// list of axis ROLES.
/// </summary>
/// <remarks>
/// <para>
/// A type may carry SEVERAL of these — one per accepted form — and validation succeeds if ANY of them
/// matches. That is not a convenience: <c>DenseLayer</c> genuinely accepts both <c>[N, Features]</c> and
/// <c>[N, Time, Features]</c>, and refusing to model that would either force false declarations or
/// require adapter layers at call sites that work correctly today.
/// </para>
/// <para>
/// <see cref="TensorAxis.Batch"/> can be marked OPTIONAL via <see cref="BatchOptional"/>, so
/// <c>[C,H,W]</c> and <c>[B,C,H,W]</c> are one declaration rather than two. Most models in this codebase
/// accept both, and writing every layout twice would double the annotation burden for no information.
/// </para>
/// <para>
/// WHAT THIS ATTRIBUTE DOES NOT CARRY: tied relationships between axes, such as a super-resolution
/// model's <c>output Width = ScaleFactor * input Width</c>. C# attributes take only compile-time
/// constants and <c>ScaleFactor</c> is a runtime option, so those live in
/// <see cref="AiDotNet.Interfaces.IShapeContract"/> instead, where they are computed from the type's
/// ACTUAL configuration and therefore cannot go stale.
/// </para>
/// <para>
/// AN ATTRIBUTE IS A CLAIM. Every annotated type gets a generated conformance test that feeds the
/// declared layout and asserts it is accepted, plus a wrong-rank case asserting it is rejected. Without
/// that, this becomes a second source of truth that can silently stop matching the code — the same
/// failure mode as a citation that no longer describes its implementation.
/// </para>
/// <para><b>For Beginners:</b> This says, in a way the compiler and the test generator can both read,
/// "this layer expects a batch of images with channels, height and width, in that order".</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, AllowMultiple = true, Inherited = true)]
public sealed class TensorLayoutAttribute : Attribute
{
    /// <summary>The axis roles, in order, outermost first.</summary>
    public TensorAxis[] Axes { get; }

    /// <summary>
    /// Whether this declaration describes the type's INPUT (the default) or its OUTPUT.
    /// </summary>
    /// <remarks>
    /// Both are needed for chain validation: the builder compares one layer's declared output against the
    /// next layer's declared input, so a type that only declares its input can be entered but not
    /// followed.
    /// </remarks>
    public TensorLayoutDirection Direction { get; set; } = TensorLayoutDirection.Input;

    /// <summary>
    /// When true, <see cref="TensorAxis.Batch"/> may be absent, so this one declaration covers both the
    /// batched and unbatched forms.
    /// </summary>
    /// <remarks>
    /// Only meaningful when the first declared axis is <see cref="TensorAxis.Batch"/>; it is ignored
    /// otherwise, because an optional axis in the middle of a layout would make rank ambiguous again and
    /// defeat the purpose.
    /// </remarks>
    public bool BatchOptional { get; set; }

    /// <summary>Optional note explaining a non-obvious layout, surfaced in validation failures.</summary>
    public string? Note { get; set; }

    /// <summary>Declares a layout as an ordered list of axis roles.</summary>
    /// <param name="axes">The axis roles, outermost first.</param>
    public TensorLayoutAttribute(params TensorAxis[] axes)
    {
        if (axes is null) throw new ArgumentNullException(nameof(axes));
        if (axes.Length == 0)
            throw new ArgumentException("A layout needs at least one axis.", nameof(axes));
        Axes = axes;
    }

    /// <summary>The rank this layout describes when the batch axis is present.</summary>
    public int Rank => Axes.Length;

    /// <summary>
    /// True when a tensor of <paramref name="rank"/> could match this declaration, allowing for an
    /// omitted optional batch axis.
    /// </summary>
    public bool AcceptsRank(int rank)
    {
        if (rank == Axes.Length) return true;

        // ONE RULE, AND THIS IS IT. The same decision was implemented a second
        // time in ShapeDeclarationValidationGenerator.Layout, and the two copies
        // had already drifted APART IN OPPOSITE DIRECTIONS: this one lacked the
        // `Axes.Length > 1` guard, so a single-axis batch-optional layout accepted
        // rank 0; the generator lacked the `Axes[0] == Batch` guard, so it
        // reported a build error for a rank the runtime would have accepted. A
        // rule duplicated in two places is a rule that will disagree with itself,
        // so the generator now calls this method instead of restating it.
        //
        // `Axes.Length > 1` rather than `> 0`: dropping the batch axis from a
        // one-axis layout leaves a rank-0 tensor, which is a scalar, not an
        // unbatched form of anything.
        return BatchOptional
            && Axes.Length > 1
            && Axes[0] == TensorAxis.Batch
            && rank == Axes.Length - 1;
    }

    /// <summary>
    /// The axis roles as they appear for a tensor of <paramref name="rank"/>, dropping the optional batch
    /// axis when the caller supplied the unbatched form. Returns null when the rank cannot match.
    /// </summary>
    public TensorAxis[]? AxesForRank(int rank)
    {
        if (rank == Axes.Length) return Axes;
        if (!AcceptsRank(rank)) return null;

        var trimmed = new TensorAxis[Axes.Length - 1];
        Array.Copy(Axes, 1, trimmed, 0, trimmed.Length);
        return trimmed;
    }

    /// <summary>Renders the layout as, for example, <c>[Batch?, Channels, Height, Width]</c>.</summary>
    public override string ToString()
    {
        var parts = new string[Axes.Length];
        for (int i = 0; i < Axes.Length; i++)
        {
            parts[i] = Axes[i].ToString();
            if (i == 0 && BatchOptional && Axes[0] == TensorAxis.Batch) parts[i] += "?";
        }
        return "[" + string.Join(", ", parts) + "]";
    }
}

/// <summary>Whether a <see cref="TensorLayoutAttribute"/> describes an input or an output.</summary>
public enum TensorLayoutDirection
{
    /// <summary>The layout the type accepts.</summary>
    Input = 0,

    /// <summary>The layout the type produces.</summary>
    Output = 1,
}
