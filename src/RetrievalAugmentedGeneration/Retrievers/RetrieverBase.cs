
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Provides a base implementation for document retrievers with common functionality.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IRetriever interface and provides common functionality
/// for document retrieval strategies. It handles validation, result limiting, and post-processing
/// while allowing derived classes to focus on implementing the core retrieval algorithm.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all retrieval methods build upon.
/// 
/// Think of it like a template for search engines:
/// - It handles common tasks (checking inputs, limiting results, sorting)
/// - Specific retrieval methods (vector search, keyword search) just fill in how they find documents
/// - This ensures all retrievers work consistently
/// </para>
/// </remarks>
public abstract class RetrieverBase<T> : IRetriever<T>
{
    private readonly int _defaultTopK;

    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the default number of documents to retrieve.
    /// </summary>
    public int DefaultTopK => _defaultTopK;

    /// <summary>
    /// Initializes a new instance of the RetrieverBase class.
    /// </summary>
    /// <param name="defaultTopK">The default number of documents to retrieve.</param>
    protected RetrieverBase(int defaultTopK = 5)
    {
        if (defaultTopK <= 0)
            throw new ArgumentException("DefaultTopK must be greater than zero", nameof(defaultTopK));

        _defaultTopK = defaultTopK;
    }

    /// <summary>
    /// Retrieves relevant documents for a given query string using the default TopK value.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <returns>A collection of relevant documents ordered by relevance (most relevant first).</returns>
    public IEnumerable<Document<T>> Retrieve(string query)
    {
        return Retrieve(query, _defaultTopK);
    }

    /// <summary>
    /// Retrieves relevant documents with a custom number of results.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <returns>A collection of relevant documents ordered by relevance (most relevant first).</returns>
    public IEnumerable<Document<T>> Retrieve(string query, int topK)
    {
        return Retrieve(query, topK, new Dictionary<string, object>());
    }

    /// <summary>
    /// Retrieves relevant documents with metadata filtering.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <param name="metadataFilters">Metadata filters to apply before retrieval.</param>
    /// <returns>A collection of filtered, relevant documents ordered by relevance.</returns>
    public IEnumerable<Document<T>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        ValidateQuery(query);
        ValidateTopK(topK);
        ValidateMetadataFilters(metadataFilters);

        var results = RetrieveCore(query, topK, metadataFilters);

        return PostProcessResults(results, topK);
    }

    /// <summary>
    /// Core retrieval logic to be implemented by derived classes.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="topK">The validated number of documents to retrieve.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>A collection of relevant documents ordered by relevance.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific retrieval algorithm.
    /// 
    /// You don't need to:
    /// - Validate the query (already done)
    /// - Validate topK (already done)
    /// - Limit results to topK (handled in PostProcessResults)
    /// - Handle null/empty inputs (already validated)
    /// 
    /// Just focus on: Finding and scoring the most relevant documents for the query.
    /// </para>
    /// </remarks>
    protected abstract IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters);

    /// <summary>
    /// Validates the query string.
    /// </summary>
    /// <param name="query">The query to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom query validation.
    /// For example, minimum length requirements or special character handling.
    /// </para>
    /// </remarks>
    protected virtual void ValidateQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));
    }

    /// <summary>
    /// Validates the topK parameter.
    /// </summary>
    /// <param name="topK">The topK value to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom topK validation.
    /// For example, maximum limits based on your retrieval system's constraints.
    /// </para>
    /// </remarks>
    protected virtual void ValidateTopK(int topK)
    {
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than zero");
    }

    /// <summary>
    /// Validates the metadata filters.
    /// </summary>
    /// <param name="metadataFilters">The metadata filters to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need to validate filter syntax
    /// or check for supported filter operations.
    /// </para>
    /// </remarks>
    protected virtual void ValidateMetadataFilters(Dictionary<string, object> metadataFilters)
    {
        if (metadataFilters == null)
            throw new ArgumentNullException(nameof(metadataFilters));
    }

    /// <summary>
    /// Post-processes retrieved results before returning them.
    /// </summary>
    /// <param name="results">The results from RetrieveCore.</param>
    /// <param name="topK">The number of results to return.</param>
    /// <returns>The processed results.</returns>
    /// <remarks>
    /// <para>
    /// The default implementation limits results to topK and ensures they're materialized.
    /// Override this to add custom post-processing like additional filtering, score normalization,
    /// or deduplication.
    /// </para>
    /// <para><b>For Implementers:</b> Override this for custom post-processing.
    /// 
    /// Common use cases:
    /// - Score normalization (ensure scores are in 0-1 range)
    /// - Deduplication (remove duplicate documents)
    /// - Diversity boosting (ensure variety in results)
    /// - Metadata enrichment (add computed fields)
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<Document<T>> PostProcessResults(IEnumerable<Document<T>> results, int topK)
    {
        return results.Take(topK).ToList();
    }
}
