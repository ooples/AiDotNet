using AiDotNet.NumericOperations;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using System;
using System.Collections.Generic;
using System.Linq;

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
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChainOfThoughtRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for reasoning (use StubGenerator or real LLM).</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    public ChainOfThoughtRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));
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

        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        // Step 1: Generate reasoning steps using LLM
        var reasoningPrompt = $@"Given the question: '{query}'

Please break this question into a chain of thought reasoning steps:
1. What are the key concepts to understand?
2. What sub-questions need to be answered?
3. In what order should information be gathered?

Provide numbered reasoning steps.";

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

    private List<string> ExtractSubQueries(string reasoning, string originalQuery)
    {
        var subQueries = new List<string> { originalQuery };

        // Simple extraction: look for numbered items or questions
        var lines = reasoning.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            
            // Extract questions (lines ending with ?)
            if (trimmed.EndsWith("?") && trimmed.Length > 10)
            {
                // Remove numbering like "1. ", "- ", etc.
                var cleaned = System.Text.RegularExpressions.Regex.Replace(trimmed, @"^[\d\.\-\*\s]+", "").Trim();
                if (cleaned.Length > 10)
                {
                    subQueries.Add(cleaned);
                }
            }
            // Extract concept mentions (lines with keywords)
            else if (trimmed.Contains("concept") || trimmed.Contains("understand") || trimmed.Contains("need"))
            {
                // Extract noun phrases after "understand" or similar verbs
                var match = System.Text.RegularExpressions.Regex.Match(trimmed, @"(?:understand|concept|about|regarding)\s+(.+?)(?:\s+\d|\s*$)", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                if (match.Success && match.Groups[1].Value.Length > 5)
                {
                    subQueries.Add($"information about {match.Groups[1].Value.Trim()}");
                }
            }
        }

        return subQueries.Distinct().Take(5).ToList();
    }
}
