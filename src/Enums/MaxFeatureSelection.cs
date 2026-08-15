namespace AiDotNet.Enums;

/// <summary>
/// How many features a tree considers at each split, for the random-forest family.
/// </summary>
/// <remarks>
/// <para>
/// Replaces a <c>string</c> option that accepted "sqrt", "log2", "all" or a parsed integer. The
/// string form had a silent failure mode: its switch ended in
/// <c>_ =&gt; (int)Math.Ceiling(Math.Sqrt(NumFeatures))</c>, so <c>"log 2"</c>, <c>"Sqrt "</c> or any
/// typo compiled, ran, and quietly trained a DIFFERENT model than the caller asked for. An enum
/// cannot be mistyped.
/// </para>
/// <para>
/// An explicit feature COUNT is expressed separately (see the owning options' <c>MaxFeatureCount</c>)
/// rather than by an enum member, because a count is a number and folding it into a name-based choice
/// is what forced the string in the first place.
/// </para>
/// <para><b>For Beginners:</b> Each tree in a forest looks at only some of the available inputs when
/// deciding how to split, which keeps the trees different from each other. This picks how many.</para>
/// </remarks>
public enum MaxFeatureSelection
{
    /// <summary>
    /// The square root of the feature count, rounded up. The default for classification forests and
    /// the value both this library and scikit-learn use.
    /// </summary>
    Sqrt,

    /// <summary>Base-2 logarithm of the feature count, rounded up.</summary>
    Log2,

    /// <summary>
    /// Every feature. Removes the feature subsampling that decorrelates the trees, so the ensemble
    /// behaves more like bagging alone.
    /// </summary>
    All
}
