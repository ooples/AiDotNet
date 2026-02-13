using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Tools;

/// <summary>
/// A production-ready tool that searches a vector database using semantic similarity.
/// Integrates with the existing IRetriever infrastructure for document retrieval.
/// </summary>
/// <remarks>
/// For Beginners:
/// This tool lets an AI agent search through documents using meaning rather than just keywords.
///
/// How it works:
/// 1. Agent decides it needs information (e.g., "What is machine learning?")
/// 2. Agent calls VectorSearchTool with the search query
/// 3. Tool uses vector database to find semantically similar documents
/// 4. Returns relevant document snippets to the agent
///
/// Unlike keyword search (which looks for exact word matches), vector search understands meaning:
/// - Search "automobile" → finds documents about "cars"
/// - Search "happy" → finds documents about "joyful", "delighted"
/// - Search "ML models" → finds documents about "machine learning algorithms"
///
/// This tool wraps your existing retriever implementations (DenseRetriever, HybridRetriever, etc.)
/// so agents can leverage your production vector database infrastructure.
///
/// Example usage in an agent:
/// <code>
/// // Setup
/// var retriever = new DenseRetriever&lt;double&gt;(vectorStore, embedder);
/// var searchTool = new VectorSearchTool&lt;double&gt;(retriever, topK: 5);
/// var agent = new Agent&lt;double&gt;(chatModel, new[] { searchTool });
///
/// // Agent automatically uses the tool when needed
/// var result = await agent.RunAsync("What are the benefits of neural networks?");
/// // Agent will search vector DB, get relevant docs, and formulate an answer
/// </code>
/// </remarks>
public class VectorSearchTool<T> : ITool
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IRetriever<T> _retriever;
    private readonly int _defaultTopK;
    private readonly bool _includeMetadata;

    /// <summary>
    /// Initializes a new instance of the <see cref="VectorSearchTool{T}"/> class.
    /// </summary>
    /// <param name="retriever">The retriever to use for searching documents.</param>
    /// <param name="topK">The default number of documents to retrieve (default: 5).</param>
    /// <param name="includeMetadata">Whether to include document metadata in results (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when retriever is null.</exception>
    /// <exception cref="ArgumentException">Thrown when topK is less than 1.</exception>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Parameters:**
    /// - retriever: Your configured vector database retriever (Dense, Hybrid, BM25, etc.)
    /// - topK: How many relevant documents to return (5 is usually good)
    /// - includeMetadata: Include extra info like source, date, author
    ///
    /// **Choosing topK:**
    /// - topK=3: Quick answers, less context
    /// - topK=5: Balanced (recommended for most cases)
    /// - topK=10: Comprehensive, more tokens, slower
    ///
    /// The retriever should already be configured with your:
    /// - Vector database connection
    /// - Embedding model
    /// - Document collection
    /// </remarks>
    public VectorSearchTool(IRetriever<T> retriever, int topK = 5, bool includeMetadata = true)
    {
        Guard.NotNull(retriever);
        _retriever = retriever;

        if (topK < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be at least 1.");
        }

        if (topK > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK cannot exceed 100 to prevent performance issues.");
        }

        _defaultTopK = topK;
        _includeMetadata = includeMetadata;
    }

    /// <inheritdoc/>
    public string Name => "VectorSearch";

    /// <inheritdoc/>
    public string Description =>
        "Searches a knowledge base using semantic similarity to find relevant information. " +
        "Input should be a natural language query describing what you're looking for. " +
        "The tool will return relevant document excerpts that may help answer the query. " +
        "You can optionally specify the number of results by adding '|topK=N' to your query. " +
        "Examples: 'machine learning algorithms', 'benefits of exercise|topK=10', 'quantum computing basics'";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Search query cannot be empty.";
        }

        try
        {
            // Parse input to extract query and optional topK
            var (query, topK) = ParseInput(input);

            // Perform the search
            var documents = _retriever.Retrieve(query, topK).ToList();

            if (documents.Count == 0)
            {
                return $"No relevant documents found for query: '{query}'";
            }

            // Format results
            return FormatResults(query, documents);
        }
        catch (Exception ex)
        {
            return $"Error performing vector search: {ex.Message}";
        }
    }

    /// <summary>
    /// Parses the input string to extract the query and optional topK parameter.
    /// </summary>
    /// <param name="input">The input string.</param>
    /// <returns>A tuple containing the query and topK value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This allows agents to customize the number of results:
    /// - "machine learning" → uses default topK
    /// - "machine learning|topK=10" → retrieves 10 results
    ///
    /// The pipe (|) separates the query from parameters.
    /// </remarks>
    private (string query, int topK) ParseInput(string input)
    {
        var parts = input.Split(new[] { '|' }, 2, StringSplitOptions.RemoveEmptyEntries);

        // Guard against malformed input (e.g., just "|" or only delimiters)
        if (parts.Length == 0 || string.IsNullOrWhiteSpace(parts[0]))
        {
            throw new ArgumentException("Search query cannot be empty or consist only of delimiters.", nameof(input));
        }

        var query = parts[0].Trim();
        var topK = _defaultTopK;

        if (parts.Length > 1)
        {
            // Try to extract topK parameter
            var paramMatch = Regex.Match(
                parts[1],
                @"topK\s*=\s*(\d+)",
                RegexOptions.IgnoreCase,
                RegexTimeout);

            if (paramMatch.Success && int.TryParse(paramMatch.Groups[1].Value, out var parsedTopK))
            {
                topK = Math.Max(1, Math.Min(parsedTopK, 20)); // Clamp between 1 and 20
            }
        }

        return (query, topK);
    }

    /// <summary>
    /// Formats the retrieved documents into a readable string.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="documents">The retrieved documents.</param>
    /// <returns>A formatted string with the search results.</returns>
    /// <remarks>
    /// For Beginners:
    /// This creates a clear, structured output that the agent can easily parse and use.
    /// Each document is numbered and includes its content and optional metadata.
    /// </remarks>
    private string FormatResults(string query, List<RetrievalAugmentedGeneration.Models.Document<T>> documents)
    {
        var result = new System.Text.StringBuilder();
        result.AppendLine($"Found {documents.Count} relevant documents for '{query}':");
        result.AppendLine();

        for (int i = 0; i < documents.Count; i++)
        {
            var doc = documents[i];
            result.AppendLine($"[{i + 1}] {doc.Content}");

            if (_includeMetadata && doc.Metadata != null && doc.Metadata.Count > 0)
            {
                var metadata = string.Join(", ", doc.Metadata.Select(kvp => $"{kvp.Key}: {kvp.Value}"));
                result.AppendLine($"    Metadata: {metadata}");
            }

            if (doc.HasRelevanceScore && doc.RelevanceScore != null)
            {
                // Format relevance score for display - supports any numeric type T
                string scoreText = doc.RelevanceScore is IFormattable formattable
                    ? formattable.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                    : doc.RelevanceScore.ToString() ?? "N/A";
                result.AppendLine($"    Relevance: {scoreText}");
            }

            result.AppendLine();
        }

        return result.ToString().TrimEnd();
    }
}
