namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for copyright and memorization detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for copyright detectors including n-gram extraction
/// and common scoring utilities. Concrete implementations provide the actual detection
/// algorithm (n-gram overlap, embedding similarity, perplexity analysis).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all copyright detectors.
/// Each detector type extends this and adds its own way of checking whether an AI
/// is copying from copyrighted content.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class CopyrightDetectorBase<T> : TextSafetyModuleBase<T>, ICopyrightDetector<T>
{
    /// <inheritdoc />
    public abstract double GetMemorizationScore(string text);

    /// <summary>
    /// Extracts character n-grams from text for overlap comparison.
    /// </summary>
    /// <param name="text">The text to extract n-grams from.</param>
    /// <param name="n">The n-gram size.</param>
    /// <returns>A set of unique n-grams found in the text.</returns>
    protected static HashSet<string> ExtractNgrams(string text, int n)
    {
        var ngrams = new HashSet<string>();
        if (text.Length < n) return ngrams;

        for (int i = 0; i <= text.Length - n; i++)
        {
            ngrams.Add(text.Substring(i, n));
        }

        return ngrams;
    }
}
