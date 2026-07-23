namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for heuristic text quality filtering.
/// </summary>
/// <remarks>
/// Applies simple rule-based filters commonly used for web-crawl cleanup (e.g., C4, CCNet).
/// </remarks>
public sealed class HeuristicTextFilterOptions
{
    /// <summary>Minimum number of words in a document. Default is 50.</summary>
    public int MinWordCount { get; set; } = 50;
    /// <summary>Maximum number of words in a document. Default is 100000.</summary>
    public int MaxWordCount { get; set; } = 100000;
    /// <summary>Minimum average word length in characters. Default is 3.</summary>
    public double MinAvgWordLength { get; set; } = 3.0;
    /// <summary>Maximum average word length in characters. Default is 20.</summary>
    public double MaxAvgWordLength { get; set; } = 20.0;
    /// <summary>Maximum ratio of special characters to total characters. Default is 0.3.</summary>
    public double MaxSpecialCharRatio { get; set; } = 0.3;
    /// <summary>Maximum ratio of uppercase characters to alphabetic characters. Default is 0.6.</summary>
    public double MaxUppercaseRatio { get; set; } = 0.6;
    /// <summary>Maximum ratio of digits to total characters. Default is 0.3.</summary>
    public double MaxDigitRatio { get; set; } = 0.3;
    /// <summary>Maximum ratio of lines that end with an ellipsis. Default is 0.3.</summary>
    public double MaxEllipsisLineRatio { get; set; } = 0.3;
    /// <summary>Minimum proportion of lines that end with punctuation. Default is 0.1.</summary>
    public double MinPunctuationEndRatio { get; set; } = 0.1;
    /// <summary>Whether to filter documents containing common boilerplate phrases. Default is true.</summary>
    public bool FilterBoilerplate { get; set; } = true;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MinWordCount < 0) throw new ArgumentOutOfRangeException(nameof(MinWordCount), "MinWordCount must be non-negative.");
        if (MaxWordCount < MinWordCount) throw new ArgumentOutOfRangeException(nameof(MaxWordCount), "MaxWordCount must be >= MinWordCount.");
        if (MinAvgWordLength < 0) throw new ArgumentOutOfRangeException(nameof(MinAvgWordLength), "MinAvgWordLength must be non-negative.");
        if (MaxAvgWordLength < MinAvgWordLength) throw new ArgumentOutOfRangeException(nameof(MaxAvgWordLength), "MaxAvgWordLength must be >= MinAvgWordLength.");
        if (MaxSpecialCharRatio < 0 || MaxSpecialCharRatio > 1) throw new ArgumentOutOfRangeException(nameof(MaxSpecialCharRatio), "MaxSpecialCharRatio must be between 0 and 1.");
        if (MaxUppercaseRatio < 0 || MaxUppercaseRatio > 1) throw new ArgumentOutOfRangeException(nameof(MaxUppercaseRatio), "MaxUppercaseRatio must be between 0 and 1.");
        if (MaxDigitRatio < 0 || MaxDigitRatio > 1) throw new ArgumentOutOfRangeException(nameof(MaxDigitRatio), "MaxDigitRatio must be between 0 and 1.");
        if (MaxEllipsisLineRatio < 0 || MaxEllipsisLineRatio > 1) throw new ArgumentOutOfRangeException(nameof(MaxEllipsisLineRatio), "MaxEllipsisLineRatio must be between 0 and 1.");
        if (MinPunctuationEndRatio < 0 || MinPunctuationEndRatio > 1) throw new ArgumentOutOfRangeException(nameof(MinPunctuationEndRatio), "MinPunctuationEndRatio must be between 0 and 1.");
    }
}
