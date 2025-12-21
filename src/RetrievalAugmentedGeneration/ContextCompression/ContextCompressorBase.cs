
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression;

/// <summary>
/// Provides a base implementation for context compressors with common functionality.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IContextCompressor interface and provides common functionality
/// for context compression strategies. It handles validation and delegates to derived classes
/// for the core compression logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all context compressors build upon.
/// 
/// Think of it like a template for reducing document size:
/// - It handles common tasks (checking inputs aren't null or empty)
/// - Specific compression methods (LLM-based, rule-based) fill in how they compress
/// - This ensures all compressors work consistently
/// </para>
/// </remarks>
public abstract class ContextCompressorBase<T> : IContextCompressor<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Compresses a collection of documents while preserving relevance to the query.
    /// </summary>
    /// <param name="documents">The documents to compress.</param>
    /// <param name="query">The query text used to determine relevance.</param>
    /// <param name="options">Optional compression parameters.</param>
    /// <returns>The compressed documents with reduced content but maintained relevance.</returns>
    public List<Document<T>> Compress(
        List<Document<T>> documents,
        string query,
        Dictionary<string, object>? options = null)
    {
        ValidateQuery(query);
        ValidateDocuments(documents);

        if (documents.Count == 0)
        {
            return new List<Document<T>>();
        }

        return CompressCore(documents, query, options);
    }

    /// <summary>
    /// Core compression logic to be implemented by derived classes.
    /// </summary>
    /// <param name="documents">The validated and non-empty list of documents.</param>
    /// <param name="query">The validated query text.</param>
    /// <param name="options">Optional compression parameters.</param>
    /// <returns>The compressed documents.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific compression algorithm.
    /// 
    /// You don't need to:
    /// - Validate the query (already done)
    /// - Validate documents (already done)
    /// - Handle null/empty inputs (already validated)
    /// 
    /// Just focus on: Compressing document content while preserving relevance to the query.
    /// </para>
    /// </remarks>
    protected abstract List<Document<T>> CompressCore(
        List<Document<T>> documents,
        string query,
        Dictionary<string, object>? options = null);

    /// <summary>
    /// Validates the query string.
    /// </summary>
    /// <param name="query">The query to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom query validation.
    /// </para>
    /// </remarks>
    protected virtual void ValidateQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));
    }

    /// <summary>
    /// Validates the document collection.
    /// </summary>
    /// <param name="documents">The documents to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom document validation.
    /// </para>
    /// </remarks>
    protected virtual void ValidateDocuments(List<Document<T>> documents)
    {
        if (documents == null)
            throw new ArgumentNullException(nameof(documents));
    }
}
