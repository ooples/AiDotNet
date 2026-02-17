namespace AiDotNet.Safety.Text;

/// <summary>
/// Detailed result from copyright and memorization detection.
/// </summary>
public class CopyrightResult
{
    /// <summary>Overall memorization score (0.0 = original, 1.0 = memorized).</summary>
    public double MemorizationScore { get; init; }

    /// <summary>Whether the content likely infringes copyright.</summary>
    public bool IsCopyrightViolation { get; init; }

    /// <summary>Detected overlapping segments with known copyrighted content.</summary>
    public IReadOnlyList<CopyrightMatch> Matches { get; init; } = Array.Empty<CopyrightMatch>();
}

/// <summary>
/// A match between generated text and potentially copyrighted content.
/// </summary>
public class CopyrightMatch
{
    /// <summary>The matching text segment.</summary>
    public string MatchedText { get; init; } = string.Empty;

    /// <summary>Start character offset in the generated text.</summary>
    public int StartIndex { get; init; }

    /// <summary>Similarity score (0.0-1.0).</summary>
    public double Similarity { get; init; }

    /// <summary>Source of the potential copyright match, if known.</summary>
    public string Source { get; init; } = string.Empty;
}
