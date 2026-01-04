using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// High-performance in-memory vector document store optimized for rapid prototyping and small-to-medium datasets.
/// </summary>
/// <remarks>
/// <para>
/// This implementation provides a pure in-memory vector store using .NET's high-performance collections,
/// offering extremely fast vector similarity search with minimal configuration. Ideal for development,
/// testing, caching layers, and applications with datasets that fit comfortably in RAM.
/// </para>
/// <para><b>For Beginners:</b> An in-memory store keeps all your data in RAM (like variables in your program).
/// 
/// Think of it like a super-fast notebook that lives in your computer's memory:
/// - Lightning-fast access (microseconds vs milliseconds)
/// - Zero configuration - just create and use
/// - Perfect for prototyping and testing
/// - Data is lost when the program stops
/// 
/// Best used for:
/// - Development and testing
/// - Session-based temporary data
/// - Caching frequently accessed vectors
/// - Small to medium datasets (< 100K documents)
/// 
/// Key characteristics:
/// - O(log n) similarity search using HNSW approximate nearest neighbor indexing
/// - Thread-safe concurrent access
/// - Low memory overhead with efficient storage
/// - Simple serialization/deserialization support
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
public class InMemoryDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly ConcurrentDictionary<string, VectorDocument<T>> _store;
    private readonly int _initialVectorDimension;
    private readonly HNSWIndex<T> _hnswIndex;
    private readonly INumericOperations<T> _numOps;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored in the in-memory store.
    /// </summary>
    public override int DocumentCount => _store.Count;

    /// <summary>
    /// Gets the dimensionality of vectors stored in this database.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the InMemoryDocumentStore class.
    /// </summary>
    /// <param name="vectorDimension">The dimension of vector embeddings.</param>
    /// <param name="databasePath">Optional logical identifier for this store instance (for debugging/logging).</param>
    /// <param name="tableName">Optional logical name for the document collection (for debugging/logging).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when vectorDimension is less than or equal to zero.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new in-memory vector store that keeps all data in RAM.
    /// 
    /// Example:
    /// <code>
    /// // Create a store with 384-dimensional embeddings
    /// var store = new InMemoryDocumentStore&lt;float&gt;(vectorDimension: 384);
    /// 
    /// // With optional identifiers for logging
    /// var store2 = new InMemoryDocumentStore&lt;float&gt;(
    ///     vectorDimension: 384,
    ///     databasePath: "session-cache",
    ///     tableName: "documents"
    /// );
    /// </code>
    /// 
    /// All data is stored in memory and will be lost when the program exits.
    /// Vector dimension must match your embedding model's output dimension.
    /// </para>
    /// </remarks>
    public InMemoryDocumentStore(int vectorDimension, string? databasePath = null, string? tableName = null)
    {
        if (vectorDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension must be positive");

        _store = new ConcurrentDictionary<string, VectorDocument<T>>();
        _initialVectorDimension = vectorDimension;
        _vectorDimension = vectorDimension;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize HNSW index for O(log n) approximate nearest neighbor search
        // Default parameters: M=16 connections, efConstruction=200, efSearch=50
        _hnswIndex = new HNSWIndex<T>(new CosineSimilarityMetric<T>(), maxConnections: 16, efConstruction: 200, efSearch: 50);
    }

    /// <summary>
    /// Core logic for adding a single vector document to the in-memory store.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    /// <remarks>
    /// <para>
    /// Stores the document in the internal dictionary with its vector embedding.
    /// Automatically initializes the vector dimension from the first document if not specified in constructor.
    /// </para>
    /// </remarks>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        _store[vectorDocument.Document.Id] = vectorDocument;
        _hnswIndex.Add(vectorDocument.Document.Id, vectorDocument.Embedding);
    }

    /// <summary>
    /// Core logic for adding multiple vector documents in a batch operation.
    /// </summary>
    /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
    /// <remarks>
    /// <para>
    /// Batch operations add all documents to the in-memory dictionary efficiently, validating
    /// that all vectors have consistent dimensionality.
    /// </para>
    /// <para><b>For Beginners:</b> Batch operations are slightly more efficient in memory stores.
    /// 
    /// Validation approach:
    /// <code>
    /// store.AddBatch(documents); // Validates all vectors have same dimension
    /// </code>
    /// 
    /// Ensures data integrity while maintaining high performance.
    /// </para>
    /// </remarks>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0)
            return;

        // Thread-safe dimension initialization on first batch  
        if (Interlocked.CompareExchange(ref _vectorDimension, vectorDocuments[0].Embedding.Length, 0) != 0)
        {
            // Dimension was already set, validate consistency
            if (_vectorDimension != vectorDocuments[0].Embedding.Length)
            {
                throw new ArgumentException(
                    $"Vector dimension mismatch. Expected {_vectorDimension}, got {vectorDocuments[0].Embedding.Length}",
                    nameof(vectorDocuments));
            }
        }

        // Validate all documents in batch and add to stores
        var hnswBatch = new Dictionary<string, Vector<T>>();
        foreach (var vd in vectorDocuments)
        {
            // Validate batch dimensions
            if (vd.Embedding.Length != _vectorDimension)
            {
                throw new ArgumentException(
                    $"Vector dimension mismatch in batch. Expected {_vectorDimension}, got {vd.Embedding.Length} for document {vd.Document.Id}",
                    nameof(vectorDocuments));
            }
            _store[vd.Document.Id] = vd;
            hnswBatch[vd.Document.Id] = vd.Embedding;
        }

        // Add all vectors to HNSW index in batch
        _hnswIndex.AddBatch(hnswBatch);
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
    /// Performs vector similarity search using in-memory calculations. In real SQLite-VSS, this would use
    /// the vss_search() SQL function to efficiently find nearest neighbors.
    /// </para>
    /// <para><b>For Beginners:</b> Finds the most similar documents using HNSW index.
    ///
    /// How it works:
    /// 1. Use HNSW graph to find approximate nearest neighbors in O(log n) time
    /// 2. Apply metadata filters to the candidates
    /// 3. Return top-k matches
    ///
    /// HNSW (Hierarchical Navigable Small World) provides ~100x faster search
    /// compared to brute-force linear scan for large datasets.
    /// 
    /// Example:
    /// <code>
    /// // Find 10 most similar documents
    /// var results = store.GetSimilar(queryVector, topK: 10);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        bool hasFilters = metadataFilters.Count > 0;

        // Use HNSW for O(log n) approximate nearest neighbor search
        // If there are metadata filters, over-fetch candidates to account for filtering
        int fetchCount = hasFilters ? Math.Min(topK * 10, _store.Count) : topK;

        if (_hnswIndex.Count == 0)
            return Enumerable.Empty<Document<T>>();

        var hnswResults = _hnswIndex.Search(queryVector, fetchCount);
        var results = new List<Document<T>>();

        foreach (var (id, score) in hnswResults)
        {
            if (!_store.TryGetValue(id, out var vd))
                continue;

            // Apply metadata filters
            if (hasFilters && !MatchesFilters(vd.Document, metadataFilters))
                continue;

            // Create a new Document instance to avoid mutating the cached one
            var newDocument = new Document<T>(vd.Document.Id, vd.Document.Content, vd.Document.Metadata)
            {
                RelevanceScore = score,
                HasRelevanceScore = true
            };
            results.Add(newDocument);

            if (results.Count >= topK)
                break;
        }

        return results;
    }

    /// <summary>
    /// Core logic for retrieving a document by its unique identifier.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gets a specific document by ID (like SQL SELECT WHERE id = ...).
    /// 
    /// Example:
    /// <code>
    /// var doc = store.GetById("doc-123");
    /// if (doc != null)
    ///     Console.WriteLine($"Found: {doc.Content}");
    /// </code>
    /// </para>
    /// </remarks>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        return _store.TryGetValue(documentId, out var vd) ? vd.Document : null;
    }

    /// <summary>
    /// Core logic for removing a document from the in-memory store.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from the internal dictionary efficiently using O(1) removal.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document from memory.
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("doc-123"))
    ///     Console.WriteLine("Document removed from memory");
    /// </code>
    /// </para>
    /// </remarks>
    protected override bool RemoveCore(string documentId)
    {
        if (_store.TryRemove(documentId, out _))
        {
            _hnswIndex.Remove(documentId);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Core logic for retrieving all documents in the database.
    /// </summary>
    /// <returns>An enumerable of all documents without their vector embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Returns all documents from the SQLite table in no particular order (like SQL SELECT * FROM table).
    /// Vector embeddings are not included in the results.
    /// </para>
    /// <para><b>For Beginners:</b> Gets every document from the database.
    /// 
    /// Use cases:
    /// - Export database contents for backup
    /// - Migrate to a different database
    /// - Bulk processing or analysis
    /// - Debugging to see all stored data
    /// 
    /// Warning: For large databases (> 10K documents), this can use significant memory.
    /// In production, consider using pagination with LIMIT and OFFSET clauses.
    /// 
    /// Example:
    /// <code>
    /// // Get all documents
    /// var allDocs = store.GetAll().ToList();
    /// Console.WriteLine($"Total documents in database: {allDocs.Count}");
    /// 
    /// // Export to JSON
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText("database_export.json", json);
    /// 
    /// // Or copy the entire SQLite file for backup
    /// File.Copy("vectors.db", "vectors_backup.db");
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        return _store.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the database and resets the vector dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears all documents from the SQLite table (like SQL DELETE FROM table or DROP TABLE) and
    /// resets the vector dimension to 0. The database file remains but is empty.
    /// </para>
    /// <para><b>For Beginners:</b> Completely empties the database.
    /// 
    /// After calling Clear():
    /// - All documents are removed (like DELETE FROM table)
    /// - Vector dimension resets to 0
    /// - Database file still exists but is empty
    /// - Ready for new documents (even with different dimensions)
    /// 
    /// Use with caution - this cannot be undone! Consider backing up the database file first.
    /// 
    /// Example:
    /// <code>
    /// // Backup before clearing
    /// File.Copy("vectors.db", "vectors_backup.db", overwrite: true);
    /// 
    /// // Clear the database
    /// store.Clear();
    /// Console.WriteLine($"Documents remaining: {store.DocumentCount}"); // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
    {
        _store.Clear();
        _hnswIndex.Clear();
        Interlocked.Exchange(ref _vectorDimension, _initialVectorDimension);
    }
}
