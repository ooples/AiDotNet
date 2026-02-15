using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for retrieving relevant documents based on a query.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// A retriever finds the most relevant documents for a given query using various
/// retrieval strategies such as dense vector search, sparse keyword matching, or
/// hybrid approaches. Implementations can range from simple vector similarity to
/// complex multi-stage retrieval pipelines.
/// </para>
/// <para><b>For Beginners:</b> A retriever is like a smart search engine for your documents.
/// 
/// Think of it like different ways to find information:
/// - Dense retrieval: Understands meaning (finds "automobile" when you search "car")
/// - Sparse retrieval: Matches keywords (finds exact words you typed)
/// - Hybrid retrieval: Combines both approaches for best results
/// 
/// When you ask a question, the retriever finds the documents most likely to contain
/// the answer, even if they don't use the exact same words you used.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Retriever")]
public interface IRetriever<T>
{
    /// <summary>
    /// Gets the default number of documents to retrieve.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many documents the retriever returns by default.
    /// 
    /// For example, if DefaultTopK = 5, the retriever will return the 5 most relevant documents
    /// unless you specifically ask for a different number.
    /// </para>
    /// </remarks>
    int DefaultTopK { get; }

    /// <summary>
    /// Retrieves relevant documents for a given query string using the default TopK value.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <returns>A collection of relevant documents ordered by relevance (most relevant first).</returns>
    /// <remarks>
    /// <para>
    /// This method finds documents relevant to the query using the retriever's configured strategy.
    /// Documents are returned in descending order of relevance, with the most relevant documents first.
    /// The number of results returned equals DefaultTopK.
    /// </para>
    /// <para><b>For Beginners:</b> This searches for documents matching your question.
    /// 
    /// For example:
    /// - Query: "How do I reset my password?"
    /// - Returns: Top 5 (or DefaultTopK) documents about password reset procedures
    /// 
    /// The results are sorted so the best match comes first, second-best second, and so on.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> Retrieve(string query);

    /// <summary>
    /// Retrieves relevant documents with a custom number of results.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <returns>A collection of relevant documents ordered by relevance (most relevant first).</returns>
    /// <remarks>
    /// <para>
    /// This overload allows specifying a custom number of results to retrieve, overriding
    /// the DefaultTopK value. This is useful when different use cases require different
    /// numbers of results.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you specify exactly how many results you want.
    /// 
    /// For example:
    /// - Retrieve("password reset", topK: 3) → Returns 3 documents
    /// - Retrieve("password reset", topK: 10) → Returns 10 documents
    /// 
    /// Use fewer results (3-5) when you need quick answers.
    /// Use more results (10-20) when you want comprehensive information.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> Retrieve(string query, int topK);

    /// <summary>
    /// Retrieves relevant documents with metadata filtering.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <param name="metadataFilters">Metadata filters to apply before retrieval.</param>
    /// <returns>A collection of filtered, relevant documents ordered by relevance.</returns>
    /// <remarks>
    /// <para>
    /// This method combines retrieval with metadata filtering, enabling queries that
    /// constrain results based on document properties. For example, retrieving only
    /// documents from a specific time period, author, or category.
    /// </para>
    /// <para><b>For Beginners:</b> This searches for documents but only looks in a specific subset.
    /// 
    /// For example:
    /// - Query: "machine learning"
    /// - Filters: { "year": 2024, "category": "research" }
    /// - Returns: Top K papers about machine learning from 2024 in the research category
    /// 
    /// Think of it like searching in a specific section of a library rather than the whole building.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters);
}
