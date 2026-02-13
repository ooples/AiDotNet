using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// A dense vector-based retriever that uses embedding similarity for document retrieval.
/// </summary>
/// <remarks>
/// <para>
/// This retriever uses vector embeddings to find semantically similar documents. It embeds
/// the query using an embedding model, then searches the document store for the most similar
/// document vectors. This approach captures semantic meaning rather than just keyword matching.
/// </para>
/// <para><b>For Beginners:</b> This retriever finds documents by meaning, not just keywords.
/// 
/// Think of it like a smart librarian who understands what you're asking:
/// - You ask: "How do cars work?"
/// - Keyword search finds: Documents with exact words "cars" and "work"
/// - Vector search finds: Documents about automobiles, engines, mechanics (similar meaning)
/// 
/// How it works:
/// 1. Convert your question to a vector (list of numbers representing meaning)
/// 2. Compare to vectors of all documents in the store
/// 3. Find documents with closest vectors (most similar meaning)
/// 4. Return the top matches
/// 
/// For example:
/// - Query: "renewable energy"
/// - Finds: Documents about solar, wind, hydroelectric (even if they don't say "renewable")
/// - Misses: Documents about fossil fuels (different meaning)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public class VectorRetriever<T> : RetrieverBase<T>
{
    private readonly IDocumentStore<T> _documentStore;
    private readonly IEmbeddingModel<T> _embeddingModel;

    /// <summary>
    /// Initializes a new instance of the VectorRetriever class.
    /// </summary>
    /// <param name="documentStore">The document store to retrieve from.</param>
    /// <param name="embeddingModel">The embedding model to use for query encoding.</param>
    /// <param name="defaultTopK">The default number of documents to retrieve.</param>
    public VectorRetriever(
        IDocumentStore<T> documentStore,
        IEmbeddingModel<T> embeddingModel,
        int defaultTopK = 5) : base(defaultTopK)
    {
        Guard.NotNull(documentStore);
        _documentStore = documentStore;
        Guard.NotNull(embeddingModel);
        _embeddingModel = embeddingModel;
    }

    /// <summary>
    /// Core retrieval logic using dense vector similarity.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="topK">The validated number of documents to retrieve.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>A collection of relevant documents ordered by relevance.</returns>
    protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        // 1. Embed the query
        var queryVector = _embeddingModel.Embed(query);

        // 2. Search document store for similar documents
        return _documentStore.GetSimilarWithFilters(queryVector, topK, metadataFilters);
    }
}
