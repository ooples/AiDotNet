global using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Weaviate-inspired document store with class-based schema organization.
/// </summary>
/// <remarks>
/// <para>
/// This implementation provides an in-memory simulation of Weaviate, a cloud-native vector database
/// that organizes data using a class-based schema system. It uses cosine similarity for retrieval.
/// </para>
/// <para><b>For Beginners:</b> Weaviate is an open-source vector database with a GraphQL API.
/// 
/// Think of classes like database tables:
/// - Each class defines a type of data (like "Article" or "Product")
/// - Documents are instances of that class
/// - Organized by schema for structured data
/// 
/// This in-memory version is good for:
/// - Prototyping Weaviate-style schemas
/// - Testing class-based organization
/// - Small to medium collections (< 100K documents)
/// 
/// Real Weaviate provides:
/// - GraphQL API for flexible queries
/// - Automatic schema inference
/// - Module system for ML models (transformers, CLIP, etc.)
/// - Multi-tenancy support
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for vector operations.</typeparam>
public class WeaviateDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly Dictionary<string, VectorDocument<T>> _documents;
    private readonly string _className;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored in the class.
    /// </summary>
    public override int DocumentCount => _documents.Count;

    /// <summary>
    /// Gets the dimensionality of vectors stored in this class.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the WeaviateDocumentStore class.
    /// </summary>
    /// <param name="className">The class name to organize documents.</param>
    /// <param name="initialCapacity">The initial capacity for the internal dictionary (default: 1000).</param>
    /// <exception cref="ArgumentException">Thrown when class name is empty or initial capacity is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new Weaviate-style document class.
    /// 
    /// Example:
    /// <code>
    /// // Create a class for articles
    /// var store = new WeaviateDocumentStore&lt;float&gt;("Article");
    /// 
    /// // Create a class for products
    /// var productStore = new WeaviateDocumentStore&lt;double&gt;("Product", 5000);
    /// </code>
    /// 
    /// The class name helps organize different types of documents.
    /// </para>
    /// </remarks>
    public WeaviateDocumentStore(string className, int initialCapacity = 1000)
    {
        if (string.IsNullOrWhiteSpace(className))
            throw new ArgumentException("Class name cannot be empty", nameof(className));
        if (initialCapacity <= 0)
            throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

        _className = className;
        _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
        _vectorDimension = 0;
    }

    /// <summary>
    /// Core logic for adding a single vector document to the class.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    /// <remarks>
    /// <para>
    /// The first document added determines the vector dimension for all documents in this class.
    /// All subsequent documents must have embeddings of the same dimension.
    /// </para>
    /// </remarks>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (_documents.Count == 0)
        {
            _vectorDimension = vectorDocument.Embedding.Length;
        }

        _documents[vectorDocument.Document.Id] = vectorDocument;
    }

    /// <summary>
    /// Core logic for adding multiple vector documents in a batch operation.
    /// </summary>
    /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
    /// <exception cref="ArgumentException">Thrown when a document's embedding has inconsistent dimensions.</exception>
    /// <remarks>
    /// <para>
    /// Batch operations are more efficient than adding documents individually.
    /// All documents must have embeddings with the same dimension as the class.
    /// </para>
    /// <para><b>For Beginners:</b> Adding many documents at once is faster.
    /// 
    /// Inefficient:
    /// <code>
    /// foreach (var doc in documents)
    ///     store.Add(doc); // Slow!
    /// </code>
    /// 
    /// Efficient:
    /// <code>
    /// store.AddBatch(documents); // Fast!
    /// </code>
    /// </para>
    /// </remarks>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (_vectorDimension == 0 && vectorDocuments.Count > 0)
        {
            _vectorDimension = vectorDocuments[0].Embedding.Length;
        }

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
    /// Core logic for similarity search using cosine similarity with optional metadata filtering.
    /// </summary>
    /// <param name="queryVector">The validated query vector.</param>
    /// <param name="topK">The validated number of documents to return.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>Top-k similar documents ordered by cosine similarity score.</returns>
    /// <remarks>
    /// <para>
    /// Performs vector similarity search across all documents in the class, optionally filtering by metadata.
    /// Results are ordered by decreasing cosine similarity.
    /// </para>
    /// <para><b>For Beginners:</b> Finds the most similar documents.
    /// 
    /// How it works:
    /// 1. Filter documents by metadata (if provided)
    /// 2. Calculate similarity between query and each document
    /// 3. Sort by similarity (highest first)
    /// 4. Return top-k matches
    /// 
    /// Example:
    /// <code>
    /// // Find 10 most similar articles
    /// var results = store.GetSimilar(queryVector, topK: 10);
    /// 
    /// // Find similar articles by specific author
    /// var filters = new Dictionary&lt;string, object&gt; { ["author"] = "John Smith" };
    /// var filtered = store.GetSimilarWithFilters(queryVector, 5, filters);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var scoredDocuments = new List<(Document<T> Document, T Score)>();

        var matchingDocuments = _documents.Values
            .Where(vectorDoc => MatchesFilters(vectorDoc.Document, metadataFilters));

        foreach (var vectorDoc in matchingDocuments)
        {
            var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vectorDoc.Embedding);
            scoredDocuments.Add((vectorDoc.Document, similarity));
        }

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
    /// Core logic for retrieving a document by its unique identifier.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gets a specific document by ID.
    /// 
    /// Example:
    /// <code>
    /// var doc = store.GetById("article-123");
    /// if (doc != null)
    ///     Console.WriteLine($"Article: {doc.Content}");
    /// </code>
    /// </para>
    /// </remarks>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
    }

    /// <summary>
    /// Core logic for removing a document from the class.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from the class. If this was the last document, the vector dimension
    /// is reset to 0, allowing a new dimension on next add.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document from the class.
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("article-123"))
    ///     Console.WriteLine("Article deleted");
    /// </code>
    /// </para>
    /// </remarks>
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
    /// Core logic for retrieving all documents in the class.
    /// </summary>
    /// <returns>An enumerable of all documents without their vector embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Returns all documents in the class in no particular order. Vector embeddings are not included.
    /// For large classes, be aware of the memory impact.
    /// </para>
    /// <para><b>For Beginners:</b> Gets every document in the class.
    /// 
    /// Use cases:
    /// - Export all documents for backup
    /// - Migrate to a different class or store
    /// - Bulk processing or analysis
    /// - Debugging to see all stored documents
    /// 
    /// Warning: For large classes (> 10K documents), this can use significant memory.
    /// 
    /// Example:
    /// <code>
    /// // Get all documents
    /// var allDocs = store.GetAll().ToList();
    /// Console.WriteLine($"Total documents in class: {allDocs.Count}");
    /// 
    /// // Export to JSON file
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText($"{_className}_export.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        return _documents.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the class and resets the vector dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears all documents from the class and resets the vector dimension to 0.
    /// The class name remains unchanged and is ready to accept new documents.
    /// </para>
    /// <para><b>For Beginners:</b> Completely empties the class.
    /// 
    /// After calling Clear():
    /// - All documents are removed
    /// - Vector dimension resets to 0
    /// - Class name stays the same
    /// - Ready for new documents (even with different dimensions)
    /// 
    /// Use with caution - this cannot be undone!
    /// 
    /// Example:
    /// <code>
    /// store.Clear();
    /// Console.WriteLine($"Documents in class: {store.DocumentCount}"); // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
    {
        _documents.Clear();
        _vectorDimension = 0;
    }
}

