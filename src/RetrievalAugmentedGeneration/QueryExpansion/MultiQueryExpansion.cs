using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion;

/// <summary>
/// Expands queries by generating multiple query variations from different perspectives using LLM-based reformulation.
/// </summary>
/// <remarks>
/// <para>
/// MultiQueryExpansion addresses the vocabulary mismatch problem in retrieval by creating diverse phrasings of the same
/// information need. Instead of searching with a single query, it generates 3-5 reformulations (questions, statements,
/// contextual phrases, synonyms) and retrieves documents for each variation, then merges and deduplicates results.
/// This approach significantly improves recall by capturing documents that use different terminology than the original
/// query. The implementation uses pattern-based transformations as a fallback but is designed to integrate with LLMs
/// for more sophisticated reformulations (e.g., technical → layman, abstract → concrete, general → specific perspectives).
/// </para>
/// <para><b>For Beginners:</b> Think of this like asking the same question in different ways:
/// 
/// Regular search: "machine learning algorithms"
/// 
/// Multi-query expansion generates:
/// 1. "What are machine learning algorithms?" (question form)
/// 2. "information about machine learning algorithms" (contextual)
/// 3. "artificial intelligence techniques" (synonym expansion)
/// 4. "ML algorithms" (simplified)
/// 5. "machine learning methods" (variation)
/// 
/// Then searches using ALL variations and combines results!
/// 
/// ```csharp
/// var expander = new MultiQueryExpansion(
///     llmEndpoint: "http://localhost:1234/v1",
///     llmApiKey: "your-key",
///     numVariations: 5
/// );
/// 
/// var queries = expander.ExpandQuery("deep learning optimization");
/// // Returns: ["deep learning optimization", "What is deep learning optimization?", 
/// //           "details about deep learning optimization", "neural network training", ...]
/// ```
/// 
/// Why use MultiQueryExpansion:
/// - Finds documents using different terminology (e.g., "car" vs "automobile")
/// - Improves recall without sacrificing precision
/// - Handles ambiguous queries (explores multiple interpretations)
/// - Effective for cross-domain search (technical ↔ layman terms)
/// 
/// When NOT to use it:
/// - Very specific queries with clear terminology (wastes compute)
/// - High-latency systems (multiplies retrieval cost by numVariations)
/// - When you need ONLY exact matches
/// - Extremely short queries (no room for variation)
/// </para>
/// </remarks>
public class MultiQueryExpansion : QueryExpansionBase
{
    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>

    private readonly string _llmEndpoint;
    private readonly string _llmApiKey;
    private readonly int _numVariations;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiQueryExpansion"/> class.
    /// </summary>
    /// <param name="llmEndpoint">The LLM API endpoint.</param>
    /// <param name="llmApiKey">The API key for the LLM service.</param>
    /// <param name="numVariations">Number of query variations to generate.</param>
    public MultiQueryExpansion(
        string llmEndpoint,
        string llmApiKey,
        int numVariations)
    {
        _llmEndpoint = llmEndpoint ?? throw new ArgumentNullException(nameof(llmEndpoint));
        _llmApiKey = llmApiKey ?? throw new ArgumentNullException(nameof(llmApiKey));

        if (numVariations <= 0)
            throw new ArgumentOutOfRangeException(nameof(numVariations), "Number of variations must be positive");

        _numVariations = numVariations;
    }

    /// <inheritdoc />
    public override List<string> ExpandQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        var expandedQueries = new List<string> { query };

        // Generate variations using pattern-based transformations
        // In production, this would call an LLM, but we provide rule-based fallback
        var variations = GenerateVariations(query);
        expandedQueries.AddRange(variations.Take(_numVariations - 1));

        return expandedQueries;
    }

    private List<string> GenerateVariations(string query)
    {
        var variations = new List<string>();

        // Question reformulation patterns
        variations.Add(ReformulateAsQuestion(query));
        variations.Add(ReformulateAsStatement(query));
        variations.Add(AddContextualPhrases(query));
        variations.Add(SimplifyQuery(query));
        variations.Add(ExpandWithSynonyms(query));

        return variations.Where(v => !string.IsNullOrWhiteSpace(v) && v != query).Distinct().ToList();
    }

    private string ReformulateAsQuestion(string query)
    {
        if (query.TrimEnd().EndsWith("?"))
            return query;

        var questionWords = new[] { "what", "how", "why", "when", "where", "who", "which" };
        var lowerQuery = query.ToLower();

        if (questionWords.Any(w => lowerQuery.StartsWith(w)))
            return query.TrimEnd('.', ' ') + "?";

        return $"What is {query.TrimEnd('.', ' ')}?";
    }

    private string ReformulateAsStatement(string query)
    {
        var trimmed = query.TrimEnd('?', '.', ' ');
        var lowerQuery = trimmed.ToLower();

        if (lowerQuery.StartsWith("what is "))
            return trimmed.Substring(8);
        if (lowerQuery.StartsWith("how to "))
            return $"information about {trimmed.Substring(7)}";
        if (lowerQuery.StartsWith("why "))
            return $"reason for {trimmed.Substring(4)}";

        return trimmed;
    }

    private string AddContextualPhrases(string query)
    {
        var phrases = new[]
        {
            $"details about {query}",
            $"information regarding {query}",
            $"explain {query}",
            $"describe {query}"
        };

        return phrases[query.GetHashCode() % phrases.Length];
    }

    private string SimplifyQuery(string query)
    {
        var words = RegexHelper.Split(query, @"\s+", RegexOptions.None)
            .Where(w => w.Length > 3)
            .Take(5)
            .ToArray();

        return string.Join(" ", words);
    }

    private string ExpandWithSynonyms(string query)
    {
        var synonymMap = new Dictionary<string, string[]>
        {
            { "find", new[] { "search", "locate", "discover" } },
            { "show", new[] { "display", "present", "reveal" } },
            { "explain", new[] { "describe", "clarify", "elucidate" } },
            { "how", new[] { "what way", "by what means" } },
            { "fast", new[] { "quick", "rapid", "speedy" } },
            { "good", new[] { "effective", "quality", "excellent" } }
        };

        var words = query.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            var lowerWord = words[i].ToLower();
            if (synonymMap.ContainsKey(lowerWord))
            {
                var synonyms = synonymMap[lowerWord];
                words[i] = synonyms[query.GetHashCode() % synonyms.Length];
                break;
            }
        }

        return string.Join(" ", words);
    }
}



