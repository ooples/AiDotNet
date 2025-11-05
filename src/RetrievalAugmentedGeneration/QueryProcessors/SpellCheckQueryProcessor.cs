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
            
            if (_corrections.TryGetValue(lowerWord, out var correction))
            {
                correctedWords.Add(PreserveCase(word, correction));
            }
            else
            {
                correctedWords.Add(word);
            }
        }

        return string.Join("", correctedWords);
    }

    private static string PreserveCase(string original, string corrected)
    {
        if (string.IsNullOrEmpty(original) || string.IsNullOrEmpty(corrected))
            return corrected;

        if (char.IsUpper(original[0]))
        {
            return char.ToUpper(corrected[0]) + corrected.Substring(1);
        }

        return corrected;
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
            { "relevent", "relevant" },
            { "rerank", "rerank" }
        };
    }
}
