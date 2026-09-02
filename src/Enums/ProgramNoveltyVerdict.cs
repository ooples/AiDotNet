namespace AiDotNet.Enums;

/// <summary>A novelty judge's answer about one candidate, including the case where it did not answer.</summary>
/// <remarks>
/// <para>
/// <see cref="Unavailable"/> exists so that "the model could not be reached, or said something unparseable" is a
/// distinct outcome rather than being folded into a verdict. The reference implementation returns <c>True</c> for
/// an empty response, an unparseable response, and any exception, which hard-codes fail-open behaviour inside the
/// judge where a caller can neither see it nor change it. Keeping the third state lets the calling policy decide,
/// and lets a run count how often the judge actually judged.
/// </para>
/// <para><b>For Beginners:</b> When you ask a language model whether two programs are meaningfully different, it
/// can say yes, say no, or fail to give a usable answer. This enumeration keeps those three cases apart so that a
/// failure is never silently counted as a yes.</para>
/// </remarks>
public enum ProgramNoveltyVerdict
{
    /// <summary>The judge produced no usable answer.</summary>
    Unavailable = 0,

    /// <summary>The judge considered the candidate meaningfully different.</summary>
    Novel = 1,

    /// <summary>The judge considered the candidate a trivial variation of the incumbent.</summary>
    NotNovel = 2
}
