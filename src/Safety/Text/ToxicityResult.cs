namespace AiDotNet.Safety.Text;

/// <summary>
/// Detailed result from toxicity detection with per-category scores and spans.
/// </summary>
public class ToxicityResult
{
    /// <summary>Overall toxicity score (0.0 = safe, 1.0 = maximally toxic).</summary>
    public double OverallScore { get; init; }

    /// <summary>Per-category toxicity scores.</summary>
    public IReadOnlyDictionary<string, double> CategoryScores { get; init; } = new Dictionary<string, double>();

    /// <summary>Detected toxic spans with their scores.</summary>
    public IReadOnlyList<ToxicSpan> ToxicSpans { get; init; } = Array.Empty<ToxicSpan>();

    /// <summary>Whether the content exceeds the configured toxicity threshold.</summary>
    public bool IsToxic { get; init; }
}

/// <summary>
/// A span of text identified as toxic.
/// </summary>
public class ToxicSpan
{
    /// <summary>Start character offset.</summary>
    public int StartIndex { get; init; }

    /// <summary>End character offset.</summary>
    public int EndIndex { get; init; }

    /// <summary>The toxic text.</summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>Toxicity score for this span.</summary>
    public double Score { get; init; }

    /// <summary>Category of toxicity detected.</summary>
    public string Category { get; init; } = string.Empty;
}
