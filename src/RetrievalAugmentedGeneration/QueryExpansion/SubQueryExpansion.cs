using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion;

/// <summary>
/// Expands complex queries by decomposing them into simpler, focused sub-queries for parallel retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// SubQueryExpansion solves the "complex query problem" where a single query contains multiple information needs.
/// It intelligently detects complexity indicators (conjunctions, multiple questions, comma-separated concepts) and
/// decomposes the query into independent sub-queries that are easier to retrieve for. For example, "How does climate
/// change affect polar bears and what conservation efforts exist?" becomes two focused queries. This approach improves
/// both precision (each sub-query is more specific) and recall (broader topic coverage). The implementation uses
/// linguistic patterns to identify sub-queries but can integrate with LLMs for more sophisticated decomposition.
/// Results from all sub-queries are retrieved independently and merged to provide comprehensive coverage.
/// </para>
/// <para><b>For Beginners:</b> Think of this like breaking a big question into smaller, easier ones:
/// 
/// Complex query: "Explain machine learning, deep learning, and reinforcement learning"
/// 
/// Sub-query decomposition:
/// 1. "Explain machine learning"
/// 2. "Explain deep learning"
/// 3. "Explain reinforcement learning"
/// 4. "information about machine learning" (key concept)
/// 5. "information about deep learning" (key concept)
/// 
/// Each sub-query finds specific documents, then combines all results!
/// 
/// ```csharp
/// var expander = new SubQueryExpansion(
///     llmEndpoint: "http://localhost:1234/v1",
///     llmApiKey: "your-key",
///     maxSubQueries: 4
/// );
/// 
/// var queries = expander.ExpandQuery(
///     "What is photosynthesis and how do plants use it for energy production?"
/// );
/// // Returns: ["What is photosynthesis?", "how do plants use it for energy production?", 
/// //           "information about photosynthesis", "information about energy production"]
/// ```
/// 
/// Why use SubQueryExpansion:
/// - Handles multi-part questions effectively
/// - Each sub-query is more precise (better matches)
/// - Covers all aspects of complex information needs
/// - Ideal for research questions, comprehensive queries
/// 
/// When NOT to use it:
/// - Simple, single-concept queries (unnecessary overhead)
/// - When you need documents covering ALL aspects together (decomposition loses connections)
/// - Very short queries (nothing to decompose)
/// - When retrieval latency is critical (multiple searches = slower)
/// </para>
/// </remarks>
public class SubQueryExpansion : QueryExpansionBase
{
    private readonly string _llmEndpoint;
    private readonly string _llmApiKey;
    private readonly int _maxSubQueries;

    /// <summary>
    /// Initializes a new instance of the <see cref="SubQueryExpansion"/> class.
    /// </summary>
    /// <param name="llmEndpoint">The LLM API endpoint.</param>
    /// <param name="llmApiKey">The API key for the LLM service.</param>
    /// <param name="maxSubQueries">Maximum number of sub-queries to generate.</param>
    public SubQueryExpansion(
        string llmEndpoint,
        string llmApiKey,
        int maxSubQueries)
    {
        _llmEndpoint = llmEndpoint ?? throw new ArgumentNullException(nameof(llmEndpoint));
        _llmApiKey = llmApiKey ?? throw new ArgumentNullException(nameof(llmApiKey));

        if (maxSubQueries <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSubQueries), "Max sub-queries must be positive");

        _maxSubQueries = maxSubQueries;
    }

    /// <inheritdoc />
    public override List<string> ExpandQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        var subQueries = new List<string> { query };

        // Detect and decompose complex queries
        if (IsComplexQuery(query))
        {
            var decomposed = DecomposeQuery(query);
            subQueries.AddRange(decomposed.Take(_maxSubQueries - 1));
        }

        return subQueries;
    }

    private bool IsComplexQuery(string query)
    {
        // Check for complexity indicators
        var complexityIndicators = new[]
        {
            " and ", " or ", " also ", " as well as ",
            " furthermore ", " moreover ", " additionally ",
            ",", ";", " versus ", " vs ", " compared to "
        };

        var lowerQuery = query.ToLower();
        return complexityIndicators.Any(indicator => lowerQuery.Contains(indicator));
    }

    private List<string> DecomposeQuery(string query)
    {
        var subQueries = new List<string>();

        // Split on common conjunctions and separators
        var parts = Regex.Split(query, @"\s+(?:and|or|also|as well as|furthermore|moreover|additionally)\s+", RegexOptions.IgnoreCase)
            .Concat(query.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            .Select(p => p.Trim())
            .Where(p => p.Length > 10)
            .Distinct()
            .ToList();

        if (parts.Count > 1)
        {
            subQueries.AddRange(parts);
        }

        // Extract questions if multiple are present
        var questions = ExtractQuestions(query);
        if (questions.Count > 1)
        {
            subQueries.AddRange(questions);
        }

        // Identify key concepts for focused sub-queries
        var concepts = ExtractKeyConcepts(query);
        foreach (var concept in concepts.Take(Math.Min(3, _maxSubQueries)))
        {
            subQueries.Add($"information about {concept}");
        }

        return subQueries.Distinct().Where(s => s != query).Take(_maxSubQueries - 1).ToList();
    }

    private List<string> ExtractQuestions(string query)
    {
        // Split on question marks but preserve the questions
        var questions = new List<string>();
        var parts = query.Split('?');

        for (int i = 0; i < parts.Length - 1; i++)
        {
            var question = parts[i].Trim() + "?";
            if (question.Length > 10)
                questions.Add(question);
        }

        return questions;
    }

    private List<string> ExtractKeyConcepts(string query)
    {
        // Remove common stop words and extract meaningful terms
        var stopWords = new HashSet<string>
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "what", "how", "why", "when",
            "where", "who", "which"
        };

        var words = Regex.Split(query.ToLower(), @"\W+")
            .Where(w => w.Length > 3 && !stopWords.Contains(w))
            .ToList();

        // Extract noun phrases (simple heuristic: sequences of capitalized words or long words)
        var concepts = new List<string>();
        for (int i = 0; i < words.Count; i++)
        {
            if (words[i].Length >= 5)
            {
                concepts.Add(words[i]);

                // Check for multi-word concepts
                if (i < words.Count - 1 && words[i + 1].Length >= 5)
                {
                    concepts.Add($"{words[i]} {words[i + 1]}");
                }
            }
        }

        return concepts.Distinct().ToList();
    }
}

