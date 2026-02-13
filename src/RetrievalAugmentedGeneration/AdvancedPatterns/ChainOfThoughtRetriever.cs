using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Chain-of-Thought retriever that generates reasoning steps before retrieving documents.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This advanced retrieval pattern uses large language models to break down complex queries
/// into intermediate reasoning steps before retrieving documents. By generating a chain of
/// thought, the retriever can identify key concepts, sub-questions, and the logical order
/// in which information should be gathered, leading to more comprehensive and relevant results.
/// </para>
/// <para><b>For Beginners:</b> Think of this like asking a research assistant to explain their thought process.
///
/// Normal retriever:
/// - Question: "How does photosynthesis impact climate change?"
/// - Action: Search for documents about "photosynthesis" and "climate change"
///
/// Chain-of-Thought retriever:
/// - Question: "How does photosynthesis impact climate change?"
/// - Reasoning: "First, I need to understand what photosynthesis is. Then, I need to know how it
///   relates to carbon dioxide. Finally, I need to connect CO2 to climate change."
/// - Actions:
///   1. Search for "what is photosynthesis"
///   2. Search for "photosynthesis carbon dioxide absorption"
///   3. Search for "CO2 levels and climate change"
/// - Result: More complete answer because we gathered all prerequisite knowledge
///
/// This is especially useful for complex questions that require understanding multiple concepts
/// in a specific order.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Create generator (StubGenerator for testing, or real LLM for production)
/// var generator = new StubGenerator&lt;double&gt;();
///
/// // Create base retriever
/// var baseRetriever = new DenseRetriever&lt;double&gt;(embeddingModel, documentStore);
///
/// // Create chain-of-thought retriever
/// var cotRetriever = new ChainOfThoughtRetriever&lt;double&gt;(generator, baseRetriever);
///
/// // Retrieve with reasoning
/// var documents = cotRetriever.Retrieve(
///     "What are the economic impacts of renewable energy adoption?",
///     topK: 10
/// );
///
/// // With self-consistency (multiple reasoning paths)
/// var documentsWithConsistency = cotRetriever.RetrieveWithSelfConsistency(
///     "What are the economic impacts of renewable energy adoption?",
///     topK: 10,
///     numPaths: 3  // Generate 3 different reasoning paths and aggregate
/// );
///
/// // The retriever will:
/// // 1. Generate reasoning steps (costs, benefits, job creation, etc.)
/// // 2. Retrieve documents for each reasoning step
/// // 3. Deduplicate and return top-10 most relevant documents
/// </code>
/// </para>
/// <para><b>Production Readiness:</b>
/// Current implementation uses IGenerator interface which can accept:
/// - StubGenerator for development/testing
/// - Real LLM (GPT-4, Claude, Gemini) for production
///
/// To make production-ready:
/// 1. Replace StubGenerator with real LLM generator
/// 2. Optionally tune the reasoning prompt for your domain
/// 3. Adjust max sub-queries limit based on LLM costs
/// 4. Consider caching reasoning for common queries
/// </para>
/// <para><b>Benefits:</b>
/// - More comprehensive results for complex queries
/// - Better coverage of prerequisite knowledge
/// - Improved relevance through structured reasoning
/// - Transparent reasoning process for debugging
/// - Self-consistency improves robustness
/// </para>
/// <para><b>Limitations:</b>
/// - Requires LLM access (costs/latency)
/// - Quality depends on LLM reasoning ability
/// - May retrieve redundant documents if reasoning overlaps
/// - Slower than direct retrieval
/// </para>
/// </remarks>
public class ChainOfThoughtRetriever<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly List<string> _fewShotExamples;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChainOfThoughtRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for reasoning (use StubGenerator or real LLM).</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    /// <param name="fewShotExamples">Optional few-shot examples for better reasoning quality.</param>
    public ChainOfThoughtRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        List<string>? fewShotExamples = null)
    {
        Guard.NotNull(generator);
        _generator = generator;
        Guard.NotNull(baseRetriever);
        _baseRetriever = baseRetriever;
        _fewShotExamples = fewShotExamples ?? new List<string>();
    }

    /// <summary>
    /// Retrieves documents using chain-of-thought reasoning.
    /// </summary>
    /// <param name="query">The user's query that requires reasoning.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <returns>Collection of documents ranked by relevance, gathered through reasoning steps.</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is not positive.</exception>
    /// <remarks>
    /// <para>
    /// This method generates intermediate reasoning steps using the LLM, extracts sub-queries
    /// from those steps, retrieves documents for each sub-query, deduplicates results, and
    /// returns the top-K most relevant documents.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method you call to get documents.
    /// 
    /// The process:
    /// 1. Send your question to an LLM to break it down into reasoning steps
    /// 2. Extract sub-questions from those steps
    /// 3. Retrieve documents for each sub-question (limit: 5 docs per sub-question)
    /// 4. Remove duplicates
    /// 5. Sort by relevance and return top-K documents
    /// 
    /// Example:
    /// - Query: "How do vaccines work and why are they important?"
    /// - Reasoning: "1. Understand immune system basics, 2. Explain vaccine mechanism, 
    ///   3. Discuss disease prevention benefits"
    /// - Sub-queries: ["immune system", "how vaccines work", "vaccine benefits"]
    /// - Retrieves: 5 docs about immune system + 5 about vaccines + 5 about benefits
    /// - Returns: Top-10 unique documents (topK=10)
    /// </para>
    /// </remarks>
    public IEnumerable<Document<T>> Retrieve(string query, int topK)
    {
        return Retrieve(query, topK, new Dictionary<string, object>());
    }

    /// <summary>
    /// Retrieves documents using chain-of-thought reasoning with metadata filtering.
    /// </summary>
    /// <param name="query">The query to retrieve documents for</param>
    /// <param name="topK">Maximum number of documents to return</param>
    /// <param name="metadataFilters">Metadata filters to apply during retrieval (e.g., tenant scoping, access control)</param>
    /// <returns>Retrieved documents sorted by relevance</returns>
    public IEnumerable<Document<T>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        // Step 1: Generate reasoning steps using LLM
        var reasoningPrompt = BuildReasoningPrompt(query, 0);
        var reasoningResponse = _generator.Generate(reasoningPrompt);

        // Step 2: Extract key concepts and sub-questions from reasoning
        var subQueries = ExtractSubQueries(reasoningResponse, query);

        // Step 3: Retrieve documents for each sub-query
        var allDocuments = new List<Document<T>>();
        var seenIds = new HashSet<string>();

        foreach (var subQuery in subQueries.Take(3)) // Limit to top 3 sub-queries
        {
            var docs = _baseRetriever.Retrieve(subQuery, topK: 5, metadataFilters); // Get 5 per sub-query with filters

            foreach (var doc in docs)
            {
                if (!seenIds.Contains(doc.Id))
                {
                    allDocuments.Add(doc);
                    seenIds.Add(doc.Id);
                }
            }
        }

        // Step 4: Return top-K documents sorted by relevance
        return allDocuments
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : default(T))
            .Take(topK);
    }

    /// <summary>
    /// Retrieves documents using self-consistency chain-of-thought reasoning.
    /// Generates multiple reasoning paths and aggregates results for improved robustness.
    /// </summary>
    /// <param name="query">The query to retrieve documents for.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <param name="numPaths">Number of different reasoning paths to generate (default: 3).</param>
    /// <param name="metadataFilters">Metadata filters to apply during retrieval.</param>
    /// <returns>Retrieved documents aggregated from multiple reasoning paths.</returns>
    /// <remarks>
    /// <para>
    /// Self-consistency improves reasoning quality by generating multiple independent
    /// reasoning chains and aggregating their results. This helps reduce the impact of
    /// any single poor reasoning path and increases overall result diversity.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of asking once, we ask the LLM to reason about
    /// the question multiple times from different angles, then combine all the documents
    /// we find. This gives us more comprehensive and reliable results.
    ///
    /// Example with numPaths=3:
    /// - Path 1 might focus on technical aspects
    /// - Path 2 might focus on practical applications
    /// - Path 3 might focus on theoretical foundations
    /// - Final result: Documents covering all three perspectives
    /// </para>
    /// </remarks>
    public IEnumerable<Document<T>> RetrieveWithSelfConsistency(
        string query,
        int topK,
        int numPaths = 3,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        if (numPaths < 1)
            throw new ArgumentOutOfRangeException(nameof(numPaths), "numPaths must be positive");

        metadataFilters ??= new Dictionary<string, object>();

        // Collect documents from multiple reasoning paths
        var allDocuments = new Dictionary<string, (Document<T> doc, int frequency)>();

        for (int i = 0; i < numPaths; i++)
        {
            // Generate reasoning with variation prompt
            var reasoningPrompt = BuildReasoningPrompt(query, i);
            var reasoningResponse = _generator.Generate(reasoningPrompt);

            // Extract sub-queries from this reasoning path
            var subQueries = ExtractSubQueries(reasoningResponse, query);

            // Retrieve documents for each sub-query
            foreach (var subQuery in subQueries.Take(3))
            {
                var docs = _baseRetriever.Retrieve(subQuery, topK: 5, metadataFilters);

                foreach (var doc in docs)
                {
                    if (allDocuments.TryGetValue(doc.Id, out var existing))
                    {
                        // Document found in multiple paths - increase frequency
                        allDocuments[doc.Id] = (existing.doc, existing.frequency + 1);
                    }
                    else
                    {
                        allDocuments[doc.Id] = (doc, 1);
                    }
                }
            }
        }

        // Rank by: 1) frequency (how many paths found it), 2) relevance score
        return allDocuments.Values
            .OrderByDescending(item => item.frequency)
            .ThenByDescending(item => item.doc.HasRelevanceScore ? item.doc.RelevanceScore : default(T))
            .Select(item => item.doc)
            .Take(topK);
    }

    /// <summary>
    /// Builds a reasoning prompt with optional few-shot examples and variation for self-consistency.
    /// </summary>
    /// <param name="query">The user's query.</param>
    /// <param name="variationIndex">Index for prompt variation (0 for standard, >0 for variations).</param>
    /// <returns>Formatted reasoning prompt.</returns>
    private string BuildReasoningPrompt(string query, int variationIndex = 0)
    {
        var promptBuilder = new System.Text.StringBuilder();

        // Add few-shot examples if provided
        if (_fewShotExamples.Count > 0)
        {
            promptBuilder.AppendLine("Here are some examples of breaking down complex questions:");
            promptBuilder.AppendLine();
            foreach (var example in _fewShotExamples)
            {
                promptBuilder.AppendLine(example);
                promptBuilder.AppendLine();
            }
        }

        promptBuilder.AppendLine($"Given the question: '{query}'");
        promptBuilder.AppendLine();

        // Vary the prompt slightly for self-consistency
        if (variationIndex == 0)
        {
            promptBuilder.AppendLine(@"Please break this question into a chain of thought reasoning steps:
1. What are the key concepts to understand?
2. What sub-questions need to be answered?
3. In what order should information be gathered?");
        }
        else if (variationIndex == 1)
        {
            promptBuilder.AppendLine(@"Analyze this question by identifying:
1. What fundamental concepts are involved?
2. What related topics should be explored?
3. How do these concepts connect to answer the question?");
        }
        else
        {
            promptBuilder.AppendLine(@"Think about this question step by step:
1. What background knowledge is needed?
2. What are the main components of this question?
3. What additional context would help answer this comprehensively?");
        }

        promptBuilder.AppendLine();
        promptBuilder.AppendLine("Provide numbered reasoning steps.");

        return promptBuilder.ToString();
    }

    /// <summary>
    /// Computes Jaro-Winkler similarity between two strings (0.0 to 1.0, where 1.0 is identical).
    /// Production-ready implementation for fuzzy string matching.
    /// </summary>
    /// <remarks>
    /// Jaro-Winkler is particularly effective for short strings and handles:
    /// - Transpositions (swapped characters)
    /// - Prefixes (rewards common starting sequences)
    /// - Typos and misspellings
    /// 
    /// Algorithm: Combines Jaro distance with prefix scaling bonus
    /// - Jaro distance considers matching characters within a window
    /// - Winkler modification adds bonus for common prefix (up to 4 chars)
    /// - Result: 1.0 = identical, 0.0 = completely different
    /// </remarks>
    private double JaroWinklerSimilarity(string s1, string s2)
    {
        if (string.IsNullOrEmpty(s1) && string.IsNullOrEmpty(s2)) return 1.0;
        if (string.IsNullOrEmpty(s1) || string.IsNullOrEmpty(s2)) return 0.0;
        if (s1 == s2) return 1.0;

        // Jaro distance calculation
        int len1 = s1.Length;
        int len2 = s2.Length;
        int matchWindow = Math.Max(len1, len2) / 2 - 1;
        if (matchWindow < 1) matchWindow = 1;

        bool[] s1Matches = new bool[len1];
        bool[] s2Matches = new bool[len2];
        int matches = 0;
        int transpositions = 0;

        // Find matches
        for (int i = 0; i < len1; i++)
        {
            int start = Math.Max(0, i - matchWindow);
            int end = Math.Min(i + matchWindow + 1, len2);

            for (int j = start; j < end; j++)
            {
                if (s2Matches[j] || s1[i] != s2[j]) continue;
                s1Matches[i] = true;
                s2Matches[j] = true;
                matches++;
                break;
            }
        }

        if (matches == 0) return 0.0;

        // Count transpositions
        int k = 0;
        for (int i = 0; i < len1; i++)
        {
            if (!s1Matches[i]) continue;
            while (!s2Matches[k]) k++;
            if (s1[i] != s2[k]) transpositions++;
            k++;
        }

        double jaro = ((double)matches / len1 + (double)matches / len2 +
                      (matches - transpositions / 2.0) / matches) / 3.0;

        // Winkler modification: add prefix bonus
        int prefixLength = 0;
        for (int i = 0; i < Math.Min(Math.Min(len1, len2), 4); i++)
        {
            if (s1[i] == s2[i]) prefixLength++;
            else break;
        }

        const double prefixScale = 0.1;
        return jaro + (prefixLength * prefixScale * (1.0 - jaro));
    }

    /// <summary>
    /// Normalizes a query string for better matching and deduplication.
    /// </summary>
    /// <remarks>
    /// Normalization steps:
    /// 1. Convert to lowercase
    /// 2. Remove extra whitespace
    /// 3. Remove common punctuation (preserve question marks for question detection)
    /// 4. Trim leading/trailing whitespace
    /// </remarks>
    private string NormalizeQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query)) return string.Empty;

        // Convert to lowercase and trim
        var normalized = query.ToLowerInvariant().Trim();

        // Remove extra whitespace
        normalized = System.Text.RegularExpressions.Regex.Replace(normalized, @"\s+", " ", System.Text.RegularExpressions.RegexOptions.None, RegexTimeout);

        // Remove punctuation except question marks
        normalized = System.Text.RegularExpressions.Regex.Replace(normalized, @"[^\w\s\?]", "", System.Text.RegularExpressions.RegexOptions.None, RegexTimeout);

        return normalized;
    }

    /// <summary>
    /// Extracts sub-queries from LLM reasoning with production-ready fuzzy deduplication.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Extraction strategy:
    /// 1. Extract explicit questions (lines ending with ?)
    /// 2. Extract concept mentions from reasoning lines
    /// 3. Normalize all extracted queries
    /// 4. Deduplicate using Jaro-Winkler fuzzy matching (threshold: 0.85)
    /// 5. Limit to top 5 unique queries
    /// </para>
    /// <para>
    /// Fuzzy deduplication prevents redundant retrievals for similar queries like:
    /// - "What is photosynthesis?" vs "what is photosynthesis"
    /// - "climate change impacts" vs "impacts of climate change"
    /// </para>
    /// </remarks>
    private List<string> ExtractSubQueries(string reasoning, string originalQuery)
    {
        var subQueries = new List<string>();
        var normalizedOriginal = NormalizeQuery(originalQuery);

        // Always include original query first
        subQueries.Add(originalQuery);

        // Extract from reasoning
        var lines = reasoning.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            // Extract explicit questions (lines ending with ?)
            if (trimmed.EndsWith("?") && trimmed.Length > 10)
            {
                // Remove numbering like "1. ", "- ", etc.
                var cleaned = System.Text.RegularExpressions.Regex.Replace(trimmed, @"^[\d\.\-\*\)\s]+", "", System.Text.RegularExpressions.RegexOptions.None, RegexTimeout).Trim();
                if (cleaned.Length > 10)
                {
                    var normalized = NormalizeQuery(cleaned);
                    if (!IsDuplicate(normalized, subQueries))
                    {
                        subQueries.Add(cleaned);
                    }
                }
            }
            // Extract concept mentions
            else if (trimmed.IndexOf("concept", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    trimmed.IndexOf("understand", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    trimmed.IndexOf("need to know", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                // Extract noun phrases after keywords
                var patterns = new[]
                {
                        @"(?:understand|concept|about|regarding|need to know)\s+(.+?)(?:\s*[\d\.]|\s*$)",
                        @"(?:what is|what are)\s+(.+?)(?:\?|$)",
                        @"key (?:concept|idea|topic)s?:\s*(.+?)(?:\s*[\d\.]|$)"
                    };

                foreach (var pattern in patterns)
                {
                    var match = System.Text.RegularExpressions.Regex.Match(
                        trimmed,
                        pattern,
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase,
                        RegexTimeout
                    );

                    if (match.Success && match.Groups[1].Value.Length > 5)
                    {
                        var extracted = match.Groups[1].Value.Trim();
                        var normalized = NormalizeQuery(extracted);

                        if (normalized.Length > 5 && !IsDuplicate(normalized, subQueries))
                        {
                            subQueries.Add($"information about {extracted}");
                        }
                    }
                }
            }
        }

        // Return top 5 unique queries
        return subQueries.Take(5).ToList();
    }

    /// <summary>
    /// Checks if a normalized query is a fuzzy duplicate of existing queries.
    /// </summary>
    /// <param name="normalizedQuery">The normalized query to check</param>
    /// <param name="existingQueries">List of existing queries to compare against</param>
    /// <returns>True if fuzzy duplicate exists (similarity >= 0.85), false otherwise</returns>
    /// <remarks>
    /// Uses Jaro-Winkler similarity with 0.85 threshold (85% similarity).
    /// This catches very similar queries while allowing reasonable variations.
    /// </remarks>
    private bool IsDuplicate(string normalizedQuery, List<string> existingQueries)
    {
        const double similarityThreshold = 0.85;

        foreach (var existing in existingQueries)
        {
            var normalizedExisting = NormalizeQuery(existing);
            var similarity = JaroWinklerSimilarity(normalizedQuery, normalizedExisting);

            if (similarity >= similarityThreshold)
            {
                return true;
            }
        }

        return false;
    }
}
