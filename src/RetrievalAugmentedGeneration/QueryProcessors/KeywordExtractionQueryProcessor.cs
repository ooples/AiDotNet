using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

// ReSharper disable once UnusedMember.Local

/// <summary>
/// Extracts key terms and phrases from queries for focused retrieval.
/// </summary>
/// <remarks>
/// <para>
/// This processor identifies and extracts the most important keywords from a query,
/// removing filler words and focusing on content-bearing terms. This helps create
/// more focused and efficient searches.
/// </para>
/// <para><b>For Beginners:</b> Picks out the important words from your question.
/// 
/// Examples:
/// - "Explain the principles of quantum entanglement in simple terms"
///   → "quantum entanglement principles simple terms"
/// - "What are the main features of the new iPhone?"
///   → "main features new iPhone"
/// - "Can you tell me about machine learning algorithms?"
///   → "machine learning algorithms"
/// 
/// This focuses your search on what really matters!
/// </para>
/// </remarks>
public class KeywordExtractionQueryProcessor : QueryProcessorBase
{
    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private readonly HashSet<string> _stopWords;
    private readonly int _minWordLength;

    /// <summary>
    /// Initializes a new instance of the KeywordExtractionQueryProcessor class.
    /// </summary>
    /// <param name="customStopWords">Optional custom set of stop words to filter out.</param>
    /// <param name="minWordLength">Minimum word length to keep (default: 2).</param>
    public KeywordExtractionQueryProcessor(
        HashSet<string>? customStopWords = null,
        int minWordLength = 2)
    {
        _stopWords = customStopWords ?? GetDefaultStopWords();
        _minWordLength = minWordLength;
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var words = Regex.Split(query.ToLowerInvariant(), @"\W+", RegexOptions.None, RegexTimeout)
            .Where(w => !string.IsNullOrWhiteSpace(w))
            .Where(w => w.Length >= _minWordLength)
            .Where(w => !_stopWords.Contains(w))
            .ToList();

        return string.Join(" ", words);
    }

    private static HashSet<string> GetDefaultStopWords()
    {
        return new HashSet<string>
        {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "this", "but", "they", "have",
            "had", "what", "when", "where", "who", "which", "why", "how",
            "can", "could", "should", "would", "may", "might", "must",
            "shall", "do", "does", "did", "tell", "me", "you",
            "about", "explain", "describe", "please"
        };
    }
}
