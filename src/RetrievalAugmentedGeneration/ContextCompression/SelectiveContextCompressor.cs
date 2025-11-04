using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression;

/// <summary>
/// Selective context compressor that picks the most relevant sentences based on the query.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Analyzes retrieved documents and selectively extracts only the sentences most relevant
/// to the query, reducing context length while preserving important information.
/// </remarks>
public class SelectiveContextCompressor<T>
{
    private readonly INumericOperations<T> _numericOperations;
    private readonly int _maxSentences;
    private readonly T _relevanceThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelectiveContextCompressor{T}"/> class.
    /// </summary>
    /// <param name="maxSentences">Maximum number of sentences to keep.</param>
    /// <param name="relevanceThreshold">Minimum relevance score to keep a sentence.</param>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public SelectiveContextCompressor(
        int maxSentences,
        T relevanceThreshold,
        INumericOperations<T> numericOperations)
    {
        if (maxSentences <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSentences), "Max sentences must be positive");
            
        _maxSentences = maxSentences;
        _relevanceThreshold = relevanceThreshold;
        _numericOperations = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
    }

    /// <summary>
    /// Compresses documents by selecting relevant sentences.
    /// </summary>
    public IEnumerable<Document<T>> Compress(string query, IEnumerable<Document<T>> documents)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (documents == null)
            throw new ArgumentNullException(nameof(documents));

        var compressed = new List<Document<T>>();

        foreach (var doc in documents)
        {
            var sentences = SplitIntoSentences(doc.Content);
            var scoredSentences = new List<(string sentence, T score)>();

            foreach (var sentence in sentences)
            {
                var score = CalculateRelevance(query, sentence);
                if (_numericOperations.GreaterThanOrEqual(score, _relevanceThreshold))
                {
                    scoredSentences.Add((sentence, score));
                }
            }

            var selectedSentences = scoredSentences
                .OrderByDescending(s => s.score)
                .Take(_maxSentences)
                .Select(s => s.sentence);

            if (selectedSentences.Any())
            {
                compressed.Add(new Document<T>
                {
                    Id = doc.Id,
                    Content = string.Join(" ", selectedSentences),
                    Metadata = doc.Metadata,
                    RelevanceScore = doc.RelevanceScore,
                    HasRelevanceScore = doc.HasRelevanceScore
                });
            }
        }

        return compressed;
    }

    private List<string> SplitIntoSentences(string text)
    {
        // Simple sentence splitting - in production would use NLP library
        return text
            .Split(new[] { ". ", "! ", "? " }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();
    }

    private T CalculateRelevance(string query, string sentence)
    {
        // Simple relevance using word overlap - in production would use embeddings
        var queryWords = query.ToLowerInvariant().Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        var sentenceWords = sentence.ToLowerInvariant().Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        
        var overlap = queryWords.Intersect(sentenceWords).Count();
        var score = overlap > 0 ? (double)overlap / queryWords.Length : 0.0;
        
        return _numericOperations.FromDouble(score);
    }
}
