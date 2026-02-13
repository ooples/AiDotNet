using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Tools;

/// <summary>
/// A production-ready tool that performs full Retrieval-Augmented Generation (RAG).
/// Combines document retrieval, reranking, and answer generation into a single tool.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations.</typeparam>
/// <remarks>
/// For Beginners:
/// RAG (Retrieval-Augmented Generation) is a powerful technique that combines:
/// 1. **Retrieval**: Finding relevant documents from your knowledge base
/// 2. **Reranking**: Sorting documents by relevance (optional but recommended)
/// 3. **Generation**: Using an LLM to answer based on retrieved documents
///
/// Why RAG is better than just using an LLM:
/// - **Grounded in facts**: Answers cite real documents, not hallucinations
/// - **Up-to-date**: Knowledge base can be updated without retraining
/// - **Verifiable**: You can check the source documents
/// - **Domain-specific**: Works with your proprietary data
///
/// This tool wraps your existing RAG infrastructure so agents can:
/// - Ask questions about your documents
/// - Get answers with citations
/// - Verify information sources
///
/// Example in an agent:
/// <code>
/// var ragTool = new RAGTool&lt;double&gt;(retriever, reranker, generator);
/// var agent = new Agent&lt;double&gt;(chatModel, new[] { ragTool });
///
/// var result = await agent.RunAsync(
///     "What are the key findings in our Q4 2023 research?");
/// // Agent uses RAGTool to search docs and generate grounded answer
/// </code>
/// </remarks>
public class RAGTool<T> : ITool
{
    private readonly IRetriever<T> _retriever;
    private readonly IReranker<T>? _reranker;
    private readonly IGenerator<T> _generator;
    private readonly int _defaultTopK;
    private readonly int? _topKAfterRerank;
    private readonly bool _includeCitations;

    /// <summary>
    /// Initializes a new instance of the <see cref="RAGTool{T}"/> class.
    /// </summary>
    /// <param name="retriever">The retriever for finding relevant documents.</param>
    /// <param name="reranker">Optional reranker for improving retrieval quality.</param>
    /// <param name="generator">The generator for creating answers from context.</param>
    /// <param name="topK">Number of documents to retrieve initially (default: 10).</param>
    /// <param name="topKAfterRerank">Number of documents after reranking (default: 5). Only used if reranker is provided.</param>
    /// <param name="includeCitations">Whether to include citations in the answer (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when retriever or generator is null.</exception>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Required components:**
    /// - retriever: Finds documents (e.g., DenseRetriever, HybridRetriever)
    /// - generator: Creates answers (e.g., OpenAI GPT, Claude)
    ///
    /// **Optional but recommended:**
    /// - reranker: Improves document ranking (better accuracy)
    ///
    /// **Configuration:**
    /// - topK: Cast a wide net initially (10 is good)
    /// - topKAfterRerank: Keep the best after reranking (5 is usually enough)
    /// - includeCitations: Show sources ([1], [2], etc.) - very useful!
    ///
    /// **Typical setup:**
    /// ```csharp
    /// var retriever = new HybridRetriever<double>(...);      // Get ~10 candidates
    /// var reranker = new CrossEncoderReranker<double>(...);  // Pick top 5
    /// var generator = new OpenAIGenerator<double>(...);      // Generate answer
    /// var ragTool = new RAGTool<double>(retriever, reranker, generator);
    /// ```
    /// </remarks>
    public RAGTool(
        IRetriever<T> retriever,
        IReranker<T>? reranker,
        IGenerator<T> generator,
        int topK = 10,
        int? topKAfterRerank = 5,
        bool includeCitations = true)
    {
        Guard.NotNull(retriever);
        _retriever = retriever;
        Guard.NotNull(generator);
        _generator = generator;

        if (topK < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be at least 1.");
        }

        if (topK > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK cannot exceed 100 to prevent performance issues.");
        }

        if (topKAfterRerank.HasValue && topKAfterRerank.Value < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(topKAfterRerank), "TopKAfterRerank must be at least 1.");
        }

        if (topKAfterRerank.HasValue && topKAfterRerank.Value > topK)
        {
            throw new ArgumentOutOfRangeException(nameof(topKAfterRerank), "TopKAfterRerank cannot exceed TopK.");
        }

        _reranker = reranker;
        _defaultTopK = topK;
        _topKAfterRerank = topKAfterRerank;
        _includeCitations = includeCitations;
    }

    /// <inheritdoc/>
    public string Name => "RAG";

    /// <inheritdoc/>
    public string Description =>
        "Searches a knowledge base and generates a grounded answer with citations. " +
        "Input should be a question or query about information that might be in the knowledge base. " +
        "The tool will find relevant documents, analyze them, and provide an answer with source citations. " +
        "This is ideal for fact-based questions where accuracy and verifiability are important. " +
        "Examples: 'What are the main features of product X?', 'Summarize our Q3 results', 'How does feature Y work?'";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Query cannot be empty.";
        }

        try
        {
            // Step 1: Retrieve relevant documents
            var retrievedDocs = _retriever.Retrieve(input, _defaultTopK).ToList();

            if (retrievedDocs.Count == 0)
            {
                return $"No relevant documents found in the knowledge base for: '{input}'";
            }

            // Step 2: Rerank if reranker is available
            IEnumerable<RetrievalAugmentedGeneration.Models.Document<T>> contextDocs = retrievedDocs;
            if (_reranker != null)
            {
                contextDocs = _reranker.Rerank(input, retrievedDocs);

                if (_topKAfterRerank.HasValue)
                {
                    contextDocs = contextDocs.Take(_topKAfterRerank.Value);
                }
            }

            var finalDocs = contextDocs.ToList();

            if (finalDocs.Count == 0)
            {
                return "No relevant documents remained after reranking.";
            }

            // Step 3: Generate answer with citations
            var groundedAnswer = _generator.GenerateGrounded(input, finalDocs);

            // Step 4: Format the response
            return FormatAnswer(input, groundedAnswer);
        }
        catch (Exception ex)
        {
            return $"Error performing RAG: {ex.Message}";
        }
    }

    /// <summary>
    /// Formats the grounded answer into a readable string for the agent.
    /// </summary>
    /// <param name="query">The original query.</param>
    /// <param name="groundedAnswer">The grounded answer from the generator.</param>
    /// <returns>A formatted string with the answer and optional citations.</returns>
    /// <remarks>
    /// For Beginners:
    /// This creates a structured output that includes:
    /// - The answer text
    /// - Citations if enabled
    /// - Confidence score if available
    /// - Source document information
    ///
    /// This makes it easy for the agent (and users) to verify the answer.
    /// </remarks>
    private string FormatAnswer(
        string query,
        RetrievalAugmentedGeneration.Models.GroundedAnswer<T> groundedAnswer)
    {
        var result = new StringBuilder();
        result.AppendLine($"Answer to '{query}':");
        result.AppendLine();
        result.AppendLine(groundedAnswer.Answer);

        if (_includeCitations && groundedAnswer.Citations != null && groundedAnswer.Citations.Count > 0)
        {
            result.AppendLine();
            result.AppendLine("Sources:");
            for (int i = 0; i < groundedAnswer.Citations.Count; i++)
            {
                var citation = groundedAnswer.Citations[i];
                result.AppendLine($"  [{i + 1}] Document ID: {citation}");
            }
        }

        if (groundedAnswer.ConfidenceScore > 0)
        {
            result.AppendLine();
            result.AppendLine($"Confidence: {groundedAnswer.ConfidenceScore:P0}");
        }

        return result.ToString().TrimEnd();
    }
}
