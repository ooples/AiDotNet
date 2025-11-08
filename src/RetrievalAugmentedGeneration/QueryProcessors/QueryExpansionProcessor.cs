namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Expands queries with synonyms and related terms to improve retrieval recall.
/// </summary>
/// <remarks>
/// <para>
/// This processor broadens the search by adding semantically similar terms to the original query.
/// This helps retrieve relevant documents that might use different terminology.
/// </para>
/// <para><b>For Beginners:</b> Adds related words to your search to find more results.
/// 
/// Examples:
/// - "AI models" → "AI models artificial intelligence machine learning models"
/// - "car" → "car automobile vehicle transportation"
/// - "photo" → "photo image picture photograph"
/// 
/// This helps you find documents even when they use different words for the same concept!
/// </para>
/// </remarks>
public class QueryExpansionProcessor : QueryProcessorBase
{
    private readonly Dictionary<string, string[]> _synonyms;
    private readonly bool _includeOriginal;

    /// <summary>
    /// Initializes a new instance of the QueryExpansionProcessor class.
    /// </summary>
    /// <param name="customSynonyms">Optional custom synonym dictionary (term → synonyms).</param>
    /// <param name="includeOriginal">Whether to include the original query terms (default: true).</param>
    public QueryExpansionProcessor(
        Dictionary<string, string[]>? customSynonyms = null,
        bool includeOriginal = true)
    {
        _synonyms = customSynonyms ?? GetDefaultSynonyms();
        _includeOriginal = includeOriginal;
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var terms = query.ToLowerInvariant().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var expandedTerms = new HashSet<string>();

        if (_includeOriginal)
        {
            foreach (var term in terms)
            {
                expandedTerms.Add(term);
            }
        }

        foreach (var term in terms)
        {
            if (_synonyms.TryGetValue(term, out var synonyms))
            {
                foreach (var synonym in synonyms)
                {
                    expandedTerms.Add(synonym);
                }
            }
        }

        return string.Join(" ", expandedTerms);
    }

    private static Dictionary<string, string[]> GetDefaultSynonyms()
    {
        return new Dictionary<string, string[]>
        {
            { "ai", new[] { "artificial intelligence", "machine learning", "deep learning" } },
            { "ml", new[] { "machine learning", "artificial intelligence" } },
            { "model", new[] { "algorithm", "network", "architecture" } },
            { "car", new[] { "automobile", "vehicle", "transportation" } },
            { "photo", new[] { "image", "picture", "photograph" } },
            { "computer", new[] { "machine", "pc", "system" } },
            { "data", new[] { "information", "dataset", "records" } },
            { "neural", new[] { "deep learning", "network" } },
            { "transformer", new[] { "attention model", "bert", "gpt" } },
            { "embedding", new[] { "vector", "representation", "encoding" } }
        };
    }
}
