using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// An in-memory document store implementation using cosine similarity for retrieval.
/// </summary>
/// <remarks>
/// <para>
/// This implementation stores all documents and their embeddings in memory using dictionaries.
/// It provides fast similarity search using cosine similarity and is suitable for development,
/// testing, and small to medium-sized document collections (up to ~100K documents).
/// For larger collections or persistent storage, consider using a dedicated vector database.
/// </para>
/// <para><b>For Beginners:</b> This is a simple document storage that keeps everything in RAM.
/// 
/// Think of it like a filing cabinet in your office:
/// - All documents are stored in memory (RAM), not on disk
/// - Very fast for searching (no database queries needed)
/// - Lost when program restarts (not persistent)
/// - Limited by available RAM
/// 
/// Good for:
/// - Development and testing
/// - Small document collections (< 100K documents)
/// - Prototyping RAG applications
/// - Unit tests
/// 
/// Not ideal for:
/// - Production with large collections (use FAISS, Milvus, etc.)
/// - When persistence is required (data survives restarts)
/// - Distributed systems (this is single-process only)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public class InMemoryDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly Dictionary<string, VectorDocument<T>> _documents;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored in the document store.
    /// </summary>
    public override int DocumentCount => _documents.Count;

    /// <summary>
    /// Gets the dimensionality of the vectors stored in this document store.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the InMemoryDocumentStore class.
    /// </summary>
    /// <param name="initialCapacity">The initial capacity for the internal dictionary (default: 1000).</param>
    public InMemoryDocumentStore(int initialCapacity = 1000)
    {
        if (initialCapacity <= 0)
            throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

        _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
        _vectorDimension = 0;
    }

    /// <summary>
    /// Removes all documents from the store.
    /// </summary>
    public override void Clear()
    {
        _documents.Clear();
        _vectorDimension = 0;
    }

    /// <summary>
    /// Core logic for adding a single vector document.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        // Set vector dimension on first add (when store is empty)
        if (_documents.Count == 0)
        {
            _vectorDimension = vectorDocument.Embedding.Length;
        }

        // Add or update document
        _documents[vectorDocument.Document.Id] = vectorDocument;
    }

    /// <summary>
    /// Core logic for adding multiple vector documents in a batch.
    /// </summary>
    /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        // Set vector dimension from first document if not set
        if (_vectorDimension == 0 && vectorDocuments.Count > 0)
        {
            _vectorDimension = vectorDocuments[0].Embedding.Length;
        }

        // Add all documents with dimension validation
        foreach (var vectorDocument in vectorDocuments)
        {
            if (vectorDocument.Embedding.Length != _vectorDimension)
                throw new ArgumentException(
                    $"Vector dimension mismatch in batch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length} for document {vectorDocument.Document.Id}",
                    nameof(vectorDocuments));

            _documents[vectorDocument.Document.Id] = vectorDocument;
        }
    }

    /// <summary>
    /// Core logic for similarity search with optional filtering.
    /// </summary>
    /// <param name="queryVector">The validated query vector.</param>
    /// <param name="topK">The validated number of documents to return.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>Top-k similar documents ordered by similarity score.</returns>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        // Calculate similarity scores for all documents
        var scoredDocuments = new List<(Document<T> Document, T Score)>();

        var matchingDocuments = _documents.Values
            .Where(vectorDoc => MatchesFilters(vectorDoc.Document, metadataFilters));

        foreach (var vectorDoc in matchingDocuments)
        {
            // Calculate cosine similarity using StatisticsHelper
            var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vectorDoc.Embedding);
            scoredDocuments.Add((vectorDoc.Document, similarity));
        }

        // Sort by similarity (descending) and take top K
        var results = scoredDocuments
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x =>
            {
                x.Document.RelevanceScore = x.Score;
                x.Document.HasRelevanceScore = true;
                return x.Document;
            })
            .ToList();

        return results;
    }

    /// <summary>
    /// Core logic for retrieving a document by ID.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
    }

    /// <summary>
    /// Core logic for removing a document by ID.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if removed; false if not found.</returns>
    protected override bool RemoveCore(string documentId)
    {
        var removed = _documents.Remove(documentId);
        if (removed && _documents.Count == 0)
        {
            _vectorDimension = 0;
        }
        return removed;
    }

    /// <summary>
    /// Gets all documents in the store.
    /// </summary>
    /// <returns>All stored documents.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns every document in the store.
    /// 
    /// Useful for:
    /// - Debugging (see what's in the store)
    /// - Exporting data
    /// - Bulk operations
    /// 
    /// Warning: Don't use this for large collections - it returns everything!
    /// </para>
    /// </remarks>
    public IEnumerable<Document<T>> GetAllDocuments()
    {
        return _documents.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Gets all vector documents in the store (including embeddings).
    /// </summary>
    /// <returns>All stored vector documents.</returns>
    /// <remarks>
    /// <para>
    /// This is useful for serialization, backup, or migrating to a different store.
    /// </para>
    /// </remarks>
    public IEnumerable<VectorDocument<T>> GetAllVectorDocuments()
    {
        return _documents.Values.ToList();
    }
}
