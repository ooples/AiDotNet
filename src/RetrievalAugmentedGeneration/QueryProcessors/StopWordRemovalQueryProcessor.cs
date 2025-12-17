using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Removes common stop words from queries to improve retrieval precision.
/// </summary>
/// <remarks>
/// <para>
/// This processor filters out common, uninformative words that don't contribute
/// to retrieval quality. By removing these words, the search becomes more focused
/// on meaningful content.
/// </para>
/// <para><b>For Beginners:</b> Removes common words that don't help with searching.
/// 
/// Examples:
/// - "What are the main features of the new iPhone?"
///   → "main features new iPhone"
/// - "How does a neural network learn from data?"
///   → "neural network learn data"
/// 
/// Words like "what", "are", "the", "of" are removed because they don't help find specific documents.
/// </para>
/// </remarks>
public class StopWordRemovalQueryProcessor : QueryProcessorBase
{
    private readonly HashSet<string> _stopWords;
    private readonly bool _preserveFirstWord;

    /// <summary>
    /// Initializes a new instance of the StopWordRemovalQueryProcessor class.
    /// </summary>
    /// <param name="customStopWords">Optional custom set of stop words.</param>
    /// <param name="preserveFirstWord">Whether to preserve the first word even if it's a stop word (default: false).</param>
    public StopWordRemovalQueryProcessor(
        HashSet<string>? customStopWords = null,
        bool preserveFirstWord = false)
    {
        _stopWords = customStopWords ?? GetDefaultStopWords();
        _preserveFirstWord = preserveFirstWord;
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var words = Regex.Split(query, @"(\s+)")
            .Where(w => !string.IsNullOrWhiteSpace(w) || w == " ")
            .ToList();

        var result = new List<string>();
        var isFirstContentWord = true;

        foreach (var word in words)
        {
            if (word == " " || string.IsNullOrWhiteSpace(word))
            {
                continue;
            }

            var lowerWord = word.ToLowerInvariant();

            if (_preserveFirstWord && isFirstContentWord)
            {
                result.Add(word);
                isFirstContentWord = false;
            }
            else if (!_stopWords.Contains(lowerWord))
            {
                result.Add(word);
                isFirstContentWord = false;
            }
        }

        return string.Join(" ", result);
    }

    private static HashSet<string> GetDefaultStopWords()
    {
        return new HashSet<string>
        {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "but", "they", "have", "had",
            "what", "when", "where", "who", "which", "why", "can", "could",
            "should", "would", "may", "might", "must", "shall", "do", "does",
            "did", "this", "these", "those", "i", "you", "me", "my", "your",
            "his", "her", "their", "our", "we", "us", "them", "she"
        };
    }
}
