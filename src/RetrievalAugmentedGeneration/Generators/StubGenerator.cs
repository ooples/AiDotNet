using System.Text.RegularExpressions;

using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Generators;

/// <summary>
/// A simple stub generator for testing and development that creates template-based answers.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This implementation creates simple grounded answers by concatenating context documents
/// with basic citation markers. It's designed for testing the RAG pipeline structure before
/// real generation models are integrated. The generator uses a template-based approach to
/// create answers that include numbered citations to source documents.
/// </para>
/// <para><b>For Beginners:</b> This is a simple placeholder until real LLM generators are ready.
/// 
/// Think of it like an auto-reply email:
/// - It doesn't actually understand the question
/// - It just formats the retrieved documents into an answer
/// - Adds citation numbers [1], [2], [3]
/// - Good enough for testing the RAG pipeline
/// - Replace with a real LLM (GPT, Claude, etc.) for production
/// 
/// For example:
/// - Question: "What is photosynthesis?"
/// - Retrieved docs: 3 biology documents
/// - Generated answer: "Based on the provided context: [Document 1 content] [1].
///   [Document 2 content] [2]. [Document 3 content] [3]."
/// 
/// Not intelligent, but proves the pipeline works!
/// This enables development on Issue #284 without waiting for transformer integration.
/// </para>
/// </remarks>
public class StubGenerator<T> : IGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly int _maxContextTokens;
    private readonly int _maxGenerationTokens;

    /// <summary>
    /// Gets the maximum number of tokens this generator can process in a single request.
    /// </summary>
    public int MaxContextTokens => _maxContextTokens;

    /// <summary>
    /// Gets the maximum number of tokens this generator can generate in a response.
    /// </summary>
    public int MaxGenerationTokens => _maxGenerationTokens;

    /// <summary>
    /// Initializes a new instance of the StubGenerator class.
    /// </summary>
    /// <param name="maxContextTokens">The maximum context tokens (default: 2048).</param>
    /// <param name="maxGenerationTokens">The maximum generation tokens (default: 500).</param>
    public StubGenerator(int maxContextTokens = 2048, int maxGenerationTokens = 500)
    {
        if (maxContextTokens <= 0)
            throw new ArgumentException("MaxContextTokens must be greater than zero", nameof(maxContextTokens));

        if (maxGenerationTokens <= 0)
            throw new ArgumentException("MaxGenerationTokens must be greater than zero", nameof(maxGenerationTokens));

        _maxContextTokens = maxContextTokens;
        _maxGenerationTokens = maxGenerationTokens;
    }

    /// <summary>
    /// Generates a text response based on a prompt.
    /// </summary>
    /// <param name="prompt">The input prompt or question.</param>
    /// <returns>The generated text response.</returns>
    public string Generate(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Prompt cannot be null or empty", nameof(prompt));

        // Return a simple query based on the prompt to simulate LLM reasoning
        // Extract key terms and formulate a search query
        if (prompt.Contains("next step", StringComparison.OrdinalIgnoreCase) ||
            prompt.Contains("what should", StringComparison.OrdinalIgnoreCase))
        {
            // This is a multi-step reasoning prompt - generate a search query
            // Extract the original query if present
            var queryMatch = Regex.Match(prompt, @"Original Query:\s*(.+?)(\n|$)", RegexOptions.IgnoreCase, RegexTimeout);
            if (queryMatch.Success)
            {
                var originalQuery = queryMatch.Groups[1].Value.Trim();
                // Validate that the extracted query is not empty
                if (!string.IsNullOrEmpty(originalQuery))
                {
                    return $"Search for more details about: {originalQuery}";
                }
            }
            return "Search for relevant information";
        }

        // For other prompts (like summarization), return a brief summary
        return "Summary of findings based on the provided context.";
    }

    /// <summary>
    /// Generates a grounded answer using provided context documents.
    /// </summary>
    /// <param name="query">The user's original query or question.</param>
    /// <param name="context">The retrieved documents providing context for the answer.</param>
    /// <returns>A grounded answer with the generated text, source documents, and extracted citations.</returns>
    public GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        // Trim and validate the extracted query to ensure it has meaningful content
        string originalQuery = query.Trim();
        if (string.IsNullOrEmpty(originalQuery))
            throw new ArgumentException("Query cannot be empty after trimming whitespace", nameof(query));

        var contextList = context?.ToList() ?? new List<Document<T>>();

        if (contextList.Count == 0)
        {
            return new GroundedAnswer<T>
            {
                Query = originalQuery,
                Answer = "I don't have enough information to answer this question.",
                SourceDocuments = new List<Document<T>>(),
                Citations = new List<string>(),
                ConfidenceScore = 0.0
            };
        }

        // Build answer with citations
        var answerBuilder = new System.Text.StringBuilder();
        answerBuilder.AppendLine($"Based on the provided context regarding '{originalQuery}':");
        answerBuilder.AppendLine();

        var citations = new List<string>();
        for (int i = 0; i < contextList.Count; i++)
        {
            var doc = contextList[i];
            var citationNum = i + 1;

            // Add a snippet of the document with citation
            var snippet = doc.Content.Length > 200
                ? doc.Content.Substring(0, 200) + "..."
                : doc.Content;

            answerBuilder.AppendLine($"{snippet} [{citationNum}]");
            answerBuilder.AppendLine();

            // Create citation reference
            citations.Add($"[{citationNum}] Document ID: {doc.Id}");
        }

        // Calculate confidence based on retrieval scores (normalized to [0,1])
        var avgScore = contextList
            .Where(d => d.HasRelevanceScore)
            .Select(d => Convert.ToDouble(d.RelevanceScore))
            .DefaultIfEmpty(0.5)
            .Average();

        // Clamp confidence score to [0,1] range (compatible with older .NET versions)
        var confidenceScore = Math.Min(1.0, Math.Max(0.0, avgScore));

        return new GroundedAnswer<T>
        {
            Query = originalQuery,
            Answer = answerBuilder.ToString().Trim(),
            SourceDocuments = contextList,
            Citations = citations,
            ConfidenceScore = confidenceScore
        };
    }
}
