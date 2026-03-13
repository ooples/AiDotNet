using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Processes queries by correcting common spelling errors using a dictionary-based approach.
/// </summary>
/// <remarks>
/// <para>
/// This processor improves retrieval accuracy by fixing typos and misspellings before
/// the query is sent to the retriever. Uses a simple edit distance algorithm combined
/// with a frequency-based dictionary to suggest corrections.
/// </para>
/// <para><b>For Beginners:</b> This fixes spelling mistakes in your search queries.
/// 
/// Examples:
/// - "photsynthesis" → "photosynthesis"
/// - "artifical intelligence" → "artificial intelligence"
/// - "machin learning" → "machine learning"
/// 
/// It helps you find documents even when you make typos!
/// </para>
/// </remarks>
public class SpellCheckQueryProcessor : QueryProcessorBase
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly Dictionary<string, string> _misspellingToCorrect;
    private readonly HashSet<string> _correctWords;
    private readonly int _maxEditDistance;

    /// <summary>
    /// Initializes a new instance of the SpellCheckQueryProcessor class.
    /// </summary>
    /// <param name="customDictionary">Optional custom dictionary of corrections (misspelling → correct spelling).</param>
    /// <param name="maxEditDistance">Maximum edit distance for fuzzy matching (default: 2).</param>
    public SpellCheckQueryProcessor(
        Dictionary<string, string>? customDictionary = null,
        int maxEditDistance = 2)
    {
        _maxEditDistance = maxEditDistance;
        _misspellingToCorrect = customDictionary ?? GetDefaultMisspellings();

        // Build set of correct words for efficient fuzzy matching
        _correctWords = new HashSet<string>(_misspellingToCorrect.Values, StringComparer.OrdinalIgnoreCase);

        // Add all correct words to the set (from common technical vocabulary)
        foreach (var word in GetCommonTechnicalWords())
        {
            _correctWords.Add(word);
        }
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var words = Regex.Split(query, @"(\s+)", RegexOptions.None, RegexTimeout);
        var correctedWords = new List<string>();

        foreach (var word in words)
        {
            if (string.IsNullOrWhiteSpace(word))
            {
                correctedWords.Add(word);
                continue;
            }

            // Extract trailing punctuation
            var punctuation = Regex.Match(word, @"[^\w]+$", RegexOptions.None, RegexTimeout).Value;
            var leadingPunctuation = Regex.Match(word, @"^[^\w]+", RegexOptions.None, RegexTimeout).Value;
            var cleanWord = word.Trim(leadingPunctuation.ToCharArray()).Trim(punctuation.ToCharArray());

            if (string.IsNullOrEmpty(cleanWord))
            {
                correctedWords.Add(word);
                continue;
            }

            var lowerWord = cleanWord.ToLowerInvariant();
            string correctedWord;

            // First try exact misspelling match
            if (_misspellingToCorrect.TryGetValue(lowerWord, out var correction))
            {
                correctedWord = PreserveCase(cleanWord, correction);
            }
            // Check if word is already correct
            else if (_correctWords.Contains(lowerWord))
            {
                correctedWord = cleanWord;
            }
            // Then try fuzzy match against correct words if enabled
            else if (_maxEditDistance > 0 && lowerWord.Length > 3)
            {
                var fuzzyMatch = FindFuzzyMatch(lowerWord);
                correctedWord = fuzzyMatch != null ? PreserveCase(cleanWord, fuzzyMatch) : cleanWord;
            }
            else
            {
                correctedWord = cleanWord;
            }

            // Reconstruct with original punctuation
            correctedWords.Add(leadingPunctuation + correctedWord + punctuation);
        }

        return string.Join("", correctedWords);
    }

    private string? FindFuzzyMatch(string word)
    {
        string? bestMatch = null;
        var minDistance = int.MaxValue;

        // Compare against correct words, not misspellings
        foreach (var correctWord in _correctWords)
        {
            // Only compare words of similar length to avoid false matches
            if (Math.Abs(correctWord.Length - word.Length) > _maxEditDistance)
                continue;

            var distance = LevenshteinDistance(word, correctWord);
            if (distance <= _maxEditDistance && distance < minDistance)
            {
                minDistance = distance;
                bestMatch = correctWord;
            }
        }

        return bestMatch;
    }

    private static int LevenshteinDistance(string source, string target)
    {
        if (string.IsNullOrEmpty(source))
            return target?.Length ?? 0;
        if (string.IsNullOrEmpty(target))
            return source.Length;

        var sourceLength = source.Length;
        var targetLength = target.Length;
        var distance = new int[sourceLength + 1, targetLength + 1];

        for (var i = 0; i <= sourceLength; i++)
            distance[i, 0] = i;
        for (var j = 0; j <= targetLength; j++)
            distance[0, j] = j;

        for (var i = 1; i <= sourceLength; i++)
        {
            for (var j = 1; j <= targetLength; j++)
            {
                var cost = source[i - 1] == target[j - 1] ? 0 : 1;
                distance[i, j] = Math.Min(
                    Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1),
                    distance[i - 1, j - 1] + cost);
            }
        }

        return distance[sourceLength, targetLength];
    }

    private static string PreserveCase(string original, string corrected)
    {
        return QueryProcessorHelpers.PreserveCase(original, corrected);
    }

    private static Dictionary<string, string> GetDefaultMisspellings()
    {
        return new Dictionary<string, string>
        {
            // Common AI/ML misspellings
            { "photsynthesis", "photosynthesis" },
            { "artifical", "artificial" },
            { "intelligance", "intelligence" },
            { "machin", "machine" },
            { "lerning", "learning" },
            { "nueral", "neural" },
            { "netowrk", "network" },
            { "algoritm", "algorithm" },
            { "optmization", "optimization" },
            { "retreival", "retrieval" },
            { "retreive", "retrieve" },
            { "genration", "generation" },
            { "embeddin", "embedding" },
            { "similrity", "similarity" },
            { "relevent", "relevant" },
            { "transformr", "transformer" },
            { "atention", "attention" },
            { "vecotr", "vector" },
            { "tensro", "tensor" },
            { "chunkin", "chunking" },
            { "semantc", "semantic" }
        };
    }

    private static HashSet<string> GetCommonTechnicalWords()
    {
        return new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            // AI/ML core terms
            "artificial", "intelligence", "machine", "learning", "neural", "network",
            "deep", "model", "algorithm", "optimization", "training", "inference",
            "embedding", "vector", "tensor", "matrix", "gradient", "backpropagation",
            
            // RAG/Search terms
            "retrieval", "augmented", "generation", "query", "document", "chunking",
            "semantic", "similarity", "relevance", "ranking", "reranking", "scoring",
            
            // Transformer/LLM terms
            "transformer", "attention", "encoder", "decoder", "tokenization", "token",
            "vocabulary", "context", "prompt", "completion", "fine-tuning",
            
            // Data structures
            "database", "index", "cache", "store", "repository", "collection",
            "graph", "tree", "list", "array", "dictionary", "hashmap",
            
            // Common verbs/modifiers
            "search", "find", "retrieve", "generate", "process", "analyze", "compute",
            "fast", "efficient", "accurate", "robust", "scalable", "distributed"
        };
    }
}
