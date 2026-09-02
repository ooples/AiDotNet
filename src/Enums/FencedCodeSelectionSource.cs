namespace AiDotNet.Enums;

/// <summary>Records which fallback level produced the code returned by a fenced-code extraction.</summary>
/// <remarks>
/// <para>
/// Extracting a full rewrite from a chat response walks a fixed ladder: a fence whose label matches the requested
/// language, then any unlabelled fence, then any other labelled fence, then the raw response text. The reference
/// implementation collapses that ladder into a single string and never says which rung it landed on, so a caller
/// cannot tell a confident extraction from a desperate one. Reporting the rung makes the confidence explicit, so a
/// pipeline can accept a labelled fence silently while asking for a retry when it had to fall back to raw prose.
/// </para>
/// <para><b>For Beginners:</b> When a chat model returns code it normally wraps it in triple backticks with a
/// language name, like <c>```python</c>. But models are inconsistent: sometimes the label is missing, sometimes it
/// is wrong, and sometimes there are no backticks at all. The extractor tries each of those possibilities in turn,
/// and this enum tells you which one actually worked. Treat <see cref="LanguageLabeledFence"/> as trustworthy and
/// <see cref="RawResponse"/> as a warning sign that the model did not follow the format.</para>
/// </remarks>
public enum FencedCodeSelectionSource
{
    /// <summary>No code could be recovered because the response was empty or whitespace only.</summary>
    None = 0,

    /// <summary>A fence whose label matched the requested language, including its accepted aliases.</summary>
    LanguageLabeledFence = 1,

    /// <summary>A fence with no language label.</summary>
    UnlabeledFence = 2,

    /// <summary>A fence labelled with some other language, chosen only because nothing better existed.</summary>
    OtherLabeledFence = 3,

    /// <summary>No fence at all; the whole response text was used verbatim.</summary>
    RawResponse = 4
}
