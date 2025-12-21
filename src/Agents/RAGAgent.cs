using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Agents;

/// <summary>
/// Implements a Retrieval-Augmented Generation (RAG) agent that answers questions by
/// retrieving relevant documents from a knowledge base, optionally reranking them,
/// and generating grounded answers with citations.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// RAGAgent is specialized for knowledge-intensive tasks where answers must be grounded
/// in a document collection (knowledge base). Unlike general agents that use tools,
/// RAGAgent has the RAG pipeline as its core reasoning mechanism.
///
/// **What is RAG?**
/// RAG = Retrieval-Augmented Generation
/// 1. **Retrieval**: Find relevant documents from your knowledge base
/// 2. **Augmentation**: Add those documents to the query as context
/// 3. **Generation**: LLM generates answer based on retrieved documents
///
/// **When to use RAGAgent:**
/// - You have a knowledge base (documents, manuals, FAQs, research papers)
/// - Questions require factual information from those documents
/// - You want answers with citations/sources
/// - Information changes frequently (update docs, not the model)
/// - You need verifiable, grounded answers
///
/// **When NOT to use RAGAgent:**
/// - For creative writing tasks
/// - When answers don't need factual grounding
/// - For simple calculation or logic tasks (use ChainOfThoughtAgent)
/// - When you need to call various tools dynamically (use Agent/ReAct)
///
/// **Example workflow:**
/// <code>
/// User: "What are the system requirements for product X?"
///
/// === RETRIEVAL ===
/// - Search knowledge base for "product X system requirements"
/// - Find 10 potentially relevant documents
///
/// === RERANKING (optional) ===
/// - Deeply analyze those 10 documents
/// - Keep top 5 most relevant
///
/// === GENERATION ===
/// - LLM reads the 5 documents
/// - Generates answer: "Product X requires [specs from doc 1],
///   with recommended [specs from doc 2]..."
/// - Includes citations: [1], [2]
///
/// Final Answer: Full answer with source citations
/// </code>
///
/// **Key benefits:**
/// - **Grounded**: Answers based on real documents, not hallucinations
/// - **Verifiable**: Citations let users check sources
/// - **Up-to-date**: Update knowledge base without retraining
/// - **Transparent**: See which documents were used
/// - **Domain-specific**: Works with your proprietary data
///
/// **Research background:**
/// Based on "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
/// which showed combining retrieval with generation significantly improves factual accuracy.
/// </remarks>
public class RAGAgent<T> : AgentBase<T>
{
    private readonly IRetriever<T> _retriever;
    private readonly IReranker<T>? _reranker;
    private readonly IGenerator<T> _generator;
    private readonly int _retrievalTopK;
    private readonly int? _rerankTopK;
    private readonly bool _includeCitations;
    private readonly bool _allowQueryRefinement;

    /// <summary>
    /// Initializes a new instance of the <see cref="RAGAgent{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model for query understanding and refinement.</param>
    /// <param name="retriever">The retriever for finding relevant documents.</param>
    /// <param name="generator">The generator for creating grounded answers.</param>
    /// <param name="reranker">Optional reranker for improving document relevance (recommended for better accuracy).</param>
    /// <param name="retrievalTopK">Number of documents to retrieve initially (default: 10).</param>
    /// <param name="rerankTopK">Number of documents to keep after reranking (default: 5). Only used if reranker provided.</param>
    /// <param name="includeCitations">Whether to include source citations in answers (default: true).</param>
    /// <param name="allowQueryRefinement">Whether to refine ambiguous queries before retrieval (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <remarks>
    /// For Beginners:
    /// Creates a RAG agent that answers questions using your knowledge base.
    ///
    /// **Required components:**
    /// - chatModel: For query understanding (e.g., OpenAI GPT, Claude)
    /// - retriever: Searches your knowledge base (e.g., DenseRetriever, HybridRetriever)
    /// - generator: Creates answers from retrieved docs (often same as chatModel)
    ///
    /// **Optional but recommended:**
    /// - reranker: Improves document ranking for better accuracy
    ///   Without reranker: Uses retriever's ranking
    ///   With reranker: Re-sorts documents more carefully
    ///
    /// **Configuration:**
    /// - retrievalTopK: Cast wide net initially (10-20 documents)
    /// - rerankTopK: Keep best after reranking (3-7 documents)
    /// - includeCitations: Show sources like [1], [2] (very useful!)
    /// - allowQueryRefinement: Clarify vague queries before searching
    ///
    /// **Query Refinement Example:**
    /// - User asks: "How do I reset it?"
    /// - Agent thinks: "Reset what? Password? Device? Settings?"
    /// - Refined query: "How do I reset the password?"
    /// - Better retrieval results!
    ///
    /// Example setup:
    /// <code>
    /// var retriever = new HybridRetriever&lt;double&gt;(...);
    /// var reranker = new CrossEncoderReranker&lt;double&gt;(...);
    /// var generator = new OpenAIGenerator&lt;double&gt;(...);
    /// var chatModel = new OpenAIChatModel&lt;double&gt;(...);
    ///
    /// var ragAgent = new RAGAgent&lt;double&gt;(
    ///     chatModel: chatModel,
    ///     retriever: retriever,
    ///     generator: generator,
    ///     reranker: reranker,
    ///     retrievalTopK: 10,      // Get 10 candidates
    ///     rerankTopK: 5,          // Keep top 5
    ///     includeCitations: true  // Show sources
    /// );
    /// </code>
    /// </remarks>
    public RAGAgent(
        IChatModel<T> chatModel,
        IRetriever<T> retriever,
        IGenerator<T> generator,
        IReranker<T>? reranker = null,
        int retrievalTopK = 10,
        int? rerankTopK = 5,
        bool includeCitations = true,
        bool allowQueryRefinement = true)
        : base(chatModel, null) // RAG agents don't use traditional tools
    {
        _retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _reranker = reranker;
        _retrievalTopK = retrievalTopK;
        _rerankTopK = rerankTopK;
        _includeCitations = includeCitations;
        _allowQueryRefinement = allowQueryRefinement;

        if (_retrievalTopK < 1)
        {
            throw new ArgumentException("Retrieval TopK must be at least 1.", nameof(retrievalTopK));
        }

        if (_rerankTopK.HasValue && _rerankTopK.Value < 1)
        {
            throw new ArgumentException("Rerank TopK must be at least 1.", nameof(rerankTopK));
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Note: The maxIterations parameter is not used by RAGAgent as it executes a single-pass
    /// retrieval-augmentation-generation pipeline rather than an iterative reasoning loop.
    /// The parameter is retained for API compatibility with the base AgentBase class.
    /// </remarks>
    public override async Task<string> RunAsync(string query, int maxIterations = 3)
    {
        // Note: maxIterations is not used in RAG agents (no iteration loop)
        // Parameter kept for base class API compatibility

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be null or whitespace.", nameof(query));
        }

        ClearScratchpad();
        AppendToScratchpad($"Original Query: {query}\n");

        // Step 1: Query refinement if enabled
        string refinedQuery = query;
        if (_allowQueryRefinement)
        {
            AppendToScratchpad("=== QUERY ANALYSIS ===");
            refinedQuery = await RefineQueryAsync(query);

            if (!string.Equals(query, refinedQuery, StringComparison.OrdinalIgnoreCase))
            {
                AppendToScratchpad($"Refined Query: {refinedQuery}");
            }
            else
            {
                AppendToScratchpad("Query is clear, no refinement needed.");
            }
            AppendToScratchpad("");
        }

        // Step 2: Retrieve relevant documents
        AppendToScratchpad("=== RETRIEVAL PHASE ===");
        AppendToScratchpad($"Searching knowledge base for top {_retrievalTopK} documents...");

        IEnumerable<Document<T>> documents;
        try
        {
            documents = _retriever.Retrieve(refinedQuery, _retrievalTopK).ToList();
        }
        catch (Exception ex)
        {
            AppendToScratchpad($"Error during retrieval: {ex.Message}");
            return $"I encountered an error searching the knowledge base: {ex.Message}";
        }

        var retrievedDocs = documents.ToList();

        if (retrievedDocs.Count == 0)
        {
            AppendToScratchpad("No relevant documents found.");
            return $"I couldn't find any relevant information in the knowledge base for: '{query}'";
        }

        AppendToScratchpad($"Retrieved {retrievedDocs.Count} documents.");
        LogDocuments("Retrieved", retrievedDocs.Take(3)); // Log first 3
        AppendToScratchpad("");

        // Step 3: Rerank if reranker available
        IEnumerable<Document<T>> contextDocs = retrievedDocs;

        if (_reranker != null)
        {
            AppendToScratchpad("=== RERANKING PHASE ===");
            AppendToScratchpad("Reranking documents for better relevance...");

            try
            {
                contextDocs = _reranker.Rerank(refinedQuery, retrievedDocs);

                if (_rerankTopK.HasValue)
                {
                    contextDocs = contextDocs.Take(_rerankTopK.Value);
                }

                var rerankedDocs = contextDocs.ToList();
                AppendToScratchpad($"Kept top {rerankedDocs.Count} documents after reranking.");
                LogDocuments("Top Reranked", rerankedDocs.Take(3));
                AppendToScratchpad("");

                contextDocs = rerankedDocs;
            }
            catch (Exception ex)
            {
                AppendToScratchpad($"Warning: Reranking failed: {ex.Message}");
                AppendToScratchpad("Continuing with original retrieval results.");
                AppendToScratchpad("");
            }
        }

        var finalDocs = contextDocs.ToList();

        if (finalDocs.Count == 0)
        {
            AppendToScratchpad("No documents remained after reranking.");
            return "I couldn't find sufficiently relevant information to answer your question.";
        }

        // Step 4: Generate grounded answer
        AppendToScratchpad("=== GENERATION PHASE ===");
        AppendToScratchpad($"Generating answer from {finalDocs.Count} context documents...");

        GroundedAnswer<T> groundedAnswer;
        try
        {
            groundedAnswer = _generator.GenerateGrounded(query, finalDocs);
        }
        catch (Exception ex)
        {
            AppendToScratchpad($"Error during generation: {ex.Message}");
            return $"I found relevant documents but encountered an error generating the answer: {ex.Message}";
        }

        AppendToScratchpad("Answer generated successfully.");
        AppendToScratchpad("");

        // Step 5: Format and return the answer
        string formattedAnswer = FormatGroundedAnswer(groundedAnswer);

        AppendToScratchpad("=== FINAL ANSWER ===");
        AppendToScratchpad(formattedAnswer);

        return formattedAnswer;
    }

    /// <summary>
    /// Refines a potentially ambiguous or unclear query into a more specific search query.
    /// </summary>
    /// <param name="query">The original query.</param>
    /// <returns>A refined query that will retrieve better results.</returns>
    /// <remarks>
    /// For Beginners:
    /// This helps clarify vague questions before searching.
    ///
    /// Examples:
    /// - "How do I reset it?" → "How do I reset the password?"
    /// - "Latest updates?" → "What are the latest product updates?"
    /// - "Installation" → "How do I install the software?"
    ///
    /// This improves retrieval accuracy by making queries more specific.
    /// </remarks>
    private async Task<string> RefineQueryAsync(string query)
    {
        var prompt = $@"Analyze this query and determine if it needs to be refined for better document retrieval.

Query: {query}

If the query is:
- Ambiguous (unclear what ""it"" or ""this"" refers to)
- Too vague (single words like ""help"" or ""installation"")
- Lacks context (pronouns without referents)

Then provide a more specific version that would work better for document search.

If the query is already clear and specific, return it unchanged.

Respond with ONLY the refined query text, no explanation.

Refined query:";

        try
        {
            var refinedQuery = await ChatModel.GenerateResponseAsync(prompt);
            var trimmedQuery = refinedQuery?.Trim() ?? string.Empty;

            // If LLM returned empty or whitespace, fall back to original query
            if (string.IsNullOrWhiteSpace(trimmedQuery))
            {
                AppendToScratchpad("Refinement returned empty response, using original query.");
                return query;
            }

            return trimmedQuery;
        }
        catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
        {
            // If refinement fails, use original query
            AppendToScratchpad($"Query refinement error: {ex.Message}");
            return query;
        }
    }

    /// <summary>
    /// Formats a grounded answer for presentation to the user.
    /// </summary>
    private string FormatGroundedAnswer(GroundedAnswer<T> groundedAnswer)
    {
        var result = new StringBuilder();
        result.AppendLine(groundedAnswer.Answer);

        if (_includeCitations && groundedAnswer.Citations != null && groundedAnswer.Citations.Count > 0)
        {
            result.AppendLine();
            result.AppendLine("Sources:");
            for (int i = 0; i < groundedAnswer.Citations.Count; i++)
            {
                result.AppendLine($"  [{i + 1}] {groundedAnswer.Citations[i]}");
            }
        }

        if (groundedAnswer.ConfidenceScore > 0)
        {
            result.AppendLine();
            result.AppendLine($"Confidence: {groundedAnswer.ConfidenceScore:P0}");
        }

        return result.ToString().TrimEnd();
    }

    /// <summary>
    /// Logs document information to the scratchpad for debugging.
    /// </summary>
    private void LogDocuments(string label, IEnumerable<Document<T>> documents)
    {
        int index = 1;
        foreach (var doc in documents)
        {
            AppendToScratchpad($"  {label} Doc {index}:");
            AppendToScratchpad($"    ID: {doc.Id}");

            if (doc.HasRelevanceScore)
            {
                AppendToScratchpad($"    Relevance: {doc.RelevanceScore}");
            }

            // Show first 100 chars of content
            var preview = doc.Content.Length > 100
                ? doc.Content.Substring(0, 100) + "..."
                : doc.Content;
            AppendToScratchpad($"    Preview: {preview}");

            index++;
        }
    }

    /// <summary>
    /// Gets a summary of the RAG pipeline configuration.
    /// </summary>
    /// <returns>A string describing the RAG pipeline setup.</returns>
    /// <remarks>
    /// For Beginners:
    /// This provides a human-readable description of how the agent is configured,
    /// useful for debugging and understanding the pipeline.
    /// </remarks>
    public string GetPipelineInfo()
    {
        var info = new StringBuilder();
        info.AppendLine("RAG Pipeline Configuration:");
        info.AppendLine($"  Retriever: {_retriever.GetType().Name}");
        info.AppendLine($"  Generator: {_generator.GetType().Name}");
        info.AppendLine($"  Reranker: {(_reranker != null ? _reranker.GetType().Name : "None")}");
        info.AppendLine($"  Retrieval TopK: {_retrievalTopK}");

        if (_rerankTopK.HasValue && _reranker != null)
        {
            info.AppendLine($"  Rerank TopK: {_rerankTopK.Value}");
        }

        info.AppendLine($"  Include Citations: {_includeCitations}");
        info.AppendLine($"  Query Refinement: {_allowQueryRefinement}");
        info.AppendLine($"  Generator Context Window: {_generator.MaxContextTokens} tokens");
        info.AppendLine($"  Generator Max Output: {_generator.MaxGenerationTokens} tokens");

        return info.ToString();
    }
}
