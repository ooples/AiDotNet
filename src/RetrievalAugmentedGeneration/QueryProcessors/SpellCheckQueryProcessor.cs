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
    private readonly Dictionary<string, string> _corrections;
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
        _corrections = customDictionary ?? GetDefaultDictionary();
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var words = Regex.Split(query, @"(\s+)");
        var correctedWords = new List<string>();

        foreach (var word in words)
        {
            if (string.IsNullOrWhiteSpace(word))
            {
                correctedWords.Add(word);
                continue;
            }

            var lowerWord = word.ToLowerInvariant();
            
            // First try exact match
            if (_corrections.TryGetValue(lowerWord, out var correction))
            {
                correctedWords.Add(PreserveCase(word, correction));
            }
            // Then try fuzzy match if enabled
            else if (_maxEditDistance > 0)
            {
                var fuzzyMatch = FindFuzzyMatch(lowerWord);
                correctedWords.Add(fuzzyMatch != null ? PreserveCase(word, fuzzyMatch) : word);
            }
            else
            {
                correctedWords.Add(word);
            }
        }

        return string.Join("", correctedWords);
    }

    private string? FindFuzzyMatch(string word)
    {
        string? bestMatch = null;
        var minDistance = int.MaxValue;

        foreach (var key in _corrections.Keys)
        {
            var distance = LevenshteinDistance(word, key);
            if (distance <= _maxEditDistance && distance < minDistance)
            {
                minDistance = distance;
                bestMatch = _corrections[key];
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

    private static Dictionary<string, string> GetDefaultDictionary()
    {
        return new Dictionary<string, string>
        {
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
            { "relevent", "relevant" }
        };
    }
}
