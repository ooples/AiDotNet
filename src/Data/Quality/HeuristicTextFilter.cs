namespace AiDotNet.Data.Quality;

/// <summary>
/// Filters text documents using simple heuristic rules for quality assessment.
/// </summary>
/// <remarks>
/// <para>
/// Applies rule-based checks common in web-crawl data cleaning (C4, CCNet, OSCAR).
/// Checks word count, character ratios, punctuation, and boilerplate phrases.
/// Fast and requires no training data.
/// </para>
/// </remarks>
public class HeuristicTextFilter
{
    private readonly HeuristicTextFilterOptions _options;

    private static readonly string[] BoilerplatePhrases =
    [
        "javascript is required",
        "cookies must be enabled",
        "sign in to continue",
        "subscribe to our newsletter",
        "terms of service",
        "privacy policy",
        "all rights reserved",
        "click here to",
        "lorem ipsum",
        "this page is",
        "page not found",
        "access denied",
        "403 forbidden",
        "404 not found",
        "under construction"
    ];

    public HeuristicTextFilter(HeuristicTextFilterOptions? options = null)
    {
        _options = options ?? new HeuristicTextFilterOptions();
    }

    /// <summary>
    /// Evaluates a single document against all heuristic rules.
    /// </summary>
    /// <param name="text">The document text.</param>
    /// <returns>True if the document passes all quality checks; false if it should be filtered out.</returns>
    public bool PassesFilter(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return false;

        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        // Word count check
        if (words.Length < _options.MinWordCount || words.Length > _options.MaxWordCount)
            return false;

        // Average word length
        double avgWordLen = words.Average(w => w.Length);
        if (avgWordLen < _options.MinAvgWordLength || avgWordLen > _options.MaxAvgWordLength)
            return false;

        // Character ratio checks
        int totalChars = text.Length;
        if (totalChars == 0) return false;

        int specialChars = 0;
        int upperChars = 0;
        int alphaChars = 0;
        int digitChars = 0;

        foreach (char c in text)
        {
            if (char.IsLetter(c))
            {
                alphaChars++;
                if (char.IsUpper(c)) upperChars++;
            }
            else if (char.IsDigit(c))
            {
                digitChars++;
            }
            else if (!char.IsWhiteSpace(c))
            {
                specialChars++;
            }
        }

        if ((double)specialChars / totalChars > _options.MaxSpecialCharRatio)
            return false;

        if (alphaChars > 0 && (double)upperChars / alphaChars > _options.MaxUppercaseRatio)
            return false;

        if ((double)digitChars / totalChars > _options.MaxDigitRatio)
            return false;

        // Line-level checks
        string[] lines = text.Split('\n');
        if (lines.Length > 0)
        {
            int ellipsisLines = lines.Count(l => l.TrimEnd().EndsWith("..."));
            if ((double)ellipsisLines / lines.Length > _options.MaxEllipsisLineRatio)
                return false;

            int punctuationEndLines = lines.Count(l =>
            {
                string trimmed = l.TrimEnd();
                return trimmed.Length > 0 && ".!?;:".Contains(trimmed[trimmed.Length - 1]);
            });
            if ((double)punctuationEndLines / lines.Length < _options.MinPunctuationEndRatio)
                return false;
        }

        // Boilerplate check
        if (_options.FilterBoilerplate)
        {
            string lower = text.ToLowerInvariant();
            foreach (string phrase in BoilerplatePhrases)
            {
                if (lower.Contains(phrase))
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Filters documents, returning indices of documents that should be removed.
    /// </summary>
    /// <param name="documents">Documents to filter.</param>
    /// <returns>Set of indices that fail the heuristic checks (should be removed).</returns>
    public HashSet<int> Filter(IReadOnlyList<string> documents)
    {
        var filtered = new HashSet<int>();

        for (int i = 0; i < documents.Count; i++)
        {
            if (!PassesFilter(documents[i]))
                filtered.Add(i);
        }

        return filtered;
    }
}
