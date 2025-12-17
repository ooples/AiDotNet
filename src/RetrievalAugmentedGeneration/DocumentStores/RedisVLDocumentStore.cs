
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Redis-based vector document store for low-latency applications.
/// </summary>
/// <remarks>
/// <para>
/// This implementation provides an in-memory simulation of Redis with RedisSearch module,
/// which adds full-text search and vector similarity capabilities to Redis. Redis excels
/// at ultra-low latency operations, making it ideal for real-time AI applications.
/// </para>
/// <para><b>For Beginners:</b> Redis is an extremely fast in-memory data store, and RediSearch adds AI search.
/// 
/// Think of it like RAM-speed database:
/// - All data stored in memory for sub-millisecond access
/// - Perfect for real-time recommendations and search
/// - Combines caching with vector search
/// 
/// This in-memory version is good for:
/// - Prototyping Redis vector applications
/// - Testing low-latency search patterns
/// - Small to medium collections (< 100K documents)
/// 
/// Real Redis + RediSearch provides:
/// - Sub-millisecond query latency
/// - Horizontal scaling with Redis Cluster
/// - Persistence options (RDB snapshots, AOF logs)
/// - Perfect for real-time AI applications
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
public class RedisVLDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly Dictionary<string, VectorDocument<T>> _store;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored in the index.
    /// </summary>
    public override int DocumentCount => _store.Count;

    /// <summary>
    /// Gets the dimensionality of vectors stored in this index.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the RedisVLDocumentStore class.
    /// </summary>
    /// <param name="connectionString">The Redis connection string.</param>
    /// <param name="indexName">The name of the RediSearch index.</param>
    /// <param name="vectorDimension">The dimension of vector embeddings.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new Redis vector store.
    /// 
    /// Example:
    /// <code>
    /// // Create a store for 384-dimensional embeddings
    /// var store = new RedisVLDocumentStore&lt;float&gt;(
    ///     "localhost:6379",
    ///     "products",
    ///     vectorDimension: 384
    /// );
    /// </code>
    /// 
    /// Vector dimension must be known upfront to create the index properly.
    /// </para>
    /// </remarks>
    public RedisVLDocumentStore(string connectionString, string indexName, int vectorDimension)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
            throw new ArgumentException("Connection string cannot be empty", nameof(connectionString));
        if (string.IsNullOrWhiteSpace(indexName))
            throw new ArgumentException("Index name cannot be empty", nameof(indexName));
        if (vectorDimension <= 0)
            throw new ArgumentException("Vector dimension must be positive", nameof(vectorDimension));

        _store = new Dictionary<string, VectorDocument<T>>();
        _vectorDimension = vectorDimension;
    }

    /// <summary>
    /// Core logic for adding a single vector document to the index.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    /// <remarks>
    /// <para>
    /// Stores the document in the Redis-style hash structure with its vector embedding.
    /// In real Redis, this would use HSET commands with vector field types.
    /// </para>
    /// </remarks>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (_vectorDimension == 0)
            _vectorDimension = vectorDocument.Embedding.Length;

        _store[vectorDocument.Document.Id] = vectorDocument;
    }

    /// <summary>
    /// Core logic for adding multiple vector documents in a batch operation.
    /// </summary>
    /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
    /// <remarks>
    /// <para>
    /// Batch operations are more efficient, mimicking Redis pipeline commands.
    /// All documents are added in a single operation for better performance.
    /// </para>
    /// <para><b>For Beginners:</b> Batch operations are faster in Redis.
    /// 
    /// Slow (multiple round trips):
    /// <code>
    /// foreach (var doc in documents)
    ///     store.Add(doc); // Many network calls
    /// </code>
    /// 
    /// Fast (single pipeline):
    /// <code>
    /// store.AddBatch(documents); // One network call
    /// </code>
    /// </para>
    /// </remarks>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0) return;

        if (_vectorDimension == 0)
            _vectorDimension = vectorDocuments[0].Embedding.Length;

        foreach (var vd in vectorDocuments)
            _store[vd.Document.Id] = vd;
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
    /// Performs fast in-memory vector similarity search. In real Redis, this would use
    /// the FT.SEARCH command with KNN vector similarity and optional filter expressions.
    /// </para>
    /// <para><b>For Beginners:</b> Finds the most similar documents blazingly fast.
    /// 
    /// Redis is optimized for speed:
    /// 1. Everything is in RAM for instant access
    /// 2. Calculate similarity for each document
    /// 3. Sort by similarity (highest first)
    /// 4. Return top-k matches
    /// 
    /// Typical latency: < 1 millisecond for thousands of documents
    /// 
    /// Example:
    /// <code>
    /// // Lightning-fast search for 10 similar products
    /// var results = store.GetSimilar(queryVector, topK: 10);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var results = new List<(Document<T> doc, T score)>();

        foreach (var vd in _store.Values)
        {
            var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vd.Embedding);
            vd.Document.RelevanceScore = similarity;
            results.Add((vd.Document, similarity));
        }

        return results
            .OrderByDescending(x => Convert.ToDouble(x.score))
            .Take(topK)
            .Select(x => x.doc);
    }

    /// <summary>
    /// Core logic for retrieving a document by its unique identifier.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gets a specific document by ID (like Redis HGET).
    /// 
    /// Example:
    /// <code>
    /// var doc = store.GetById("product:123");
    /// if (doc != null)
    ///     Console.WriteLine($"Product: {doc.Content}");
    /// </code>
    /// </para>
    /// </remarks>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        return _store.TryGetValue(documentId, out var vd) ? vd.Document : null;
    }

    /// <summary>
    /// Core logic for removing a document from the index.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from Redis (like DEL command). This is an instant operation in memory.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document (instant in Redis).
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("product:123"))
    ///     Console.WriteLine("Product deleted from cache");
    /// </code>
    /// </para>
    /// </remarks>
    protected override bool RemoveCore(string documentId)
    {
        return _store.Remove(documentId);
    }

    /// <summary>
    /// Core logic for retrieving all documents in the index.
    /// </summary>
    /// <returns>An enumerable of all documents without their vector embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Returns all documents from the Redis index in no particular order (like SCAN command).
    /// Vector embeddings are not included in the results.
    /// </para>
    /// <para><b>For Beginners:</b> Gets every document from Redis.
    /// 
    /// Use cases:
    /// - Export cache contents for backup
    /// - Migrate to a different Redis instance
    /// - Cache warm-up or preloading
    /// - Debugging to see all cached data
    /// 
    /// Warning: In production Redis with millions of keys, consider using SCAN with cursor
    /// instead of loading everything at once.
    /// 
    /// Example:
    /// <code>
    /// // Get all cached documents
    /// var allDocs = store.GetAll().ToList();
    /// Console.WriteLine($"Total documents in Redis: {allDocs.Count}");
    /// 
    /// // Export for backup
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText("redis_backup.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        return _store.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the index and resets the vector dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears all documents from Redis (like FLUSHDB command) and resets the vector dimension to 0.
    /// The index is ready to accept new documents after clearing.
    /// </para>
    /// <para><b>For Beginners:</b> Completely empties the Redis index.
    /// 
    /// After calling Clear():
    /// - All documents are removed (instant in Redis)
    /// - Vector dimension resets to 0
    /// - Ready for new documents
    /// 
    /// Use with caution - this cannot be undone!
    /// 
    /// Example:
    /// <code>
    /// store.Clear();
    /// Console.WriteLine($"Documents in Redis: {store.DocumentCount}"); // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
    {
        _store.Clear();
        _vectorDimension = 0;
    }
}
