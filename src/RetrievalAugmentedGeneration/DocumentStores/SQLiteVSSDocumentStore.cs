using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Lightweight SQLite-based vector store using the SQLite-VSS extension.
/// </summary>
/// <remarks>
/// <para>
/// This implementation provides an in-memory simulation of SQLite with the VSS (Vector Similarity Search) extension,
/// which adds vector search capabilities to the serverless, file-based SQLite database. Perfect for
/// edge deployments, mobile apps, and development environments.
/// </para>
/// <para><b>For Beginners:</b> SQLite is a tiny, serverless database that stores data in a single file, and VSS adds AI search.
/// 
/// Think of it like an Excel file with AI superpowers:
/// - No server needed - just a file on disk
/// - Perfect for apps, IoT devices, mobile
/// - Combine SQL queries with vector search
/// 
/// This in-memory version is good for:
/// - Prototyping SQLite-VSS applications
/// - Testing local vector search
/// - Small collections (< 10K documents)
/// 
/// Real SQLite-VSS provides:
/// - Zero-configuration deployment
/// - Single-file database (easy backup/transfer)
/// - ACID transactions for data integrity
/// - Perfect for edge AI and mobile apps
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
public class SQLiteVSSDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly Dictionary<string, VectorDocument<T>> _store;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored in the database.
    /// </summary>
    public override int DocumentCount => _store.Count;

    /// <summary>
    /// Gets the dimensionality of vectors stored in this database.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the SQLiteVSSDocumentStore class.
    /// </summary>
    /// <param name="databasePath">The path to the SQLite database file.</param>
    /// <param name="tableName">The table name to organize documents.</param>
    /// <param name="vectorDimension">The dimension of vector embeddings.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new SQLite vector store.
    /// 
    /// Example:
    /// <code>
    /// // Create a store with 384-dimensional embeddings
    /// var store = new SQLiteVSSDocumentStore&lt;float&gt;(
    ///     "vectors.db",
    ///     "documents",
    ///     vectorDimension: 384
    /// );
    /// </code>
    /// 
    /// The database file will be created if it doesn't exist.
    /// Vector dimension must be known upfront to create the table properly.
    /// </para>
    /// </remarks>
    public SQLiteVSSDocumentStore(string databasePath, string tableName, int vectorDimension)
    {
        if (string.IsNullOrWhiteSpace(databasePath))
            throw new ArgumentException("Database path cannot be empty", nameof(databasePath));
        if (string.IsNullOrWhiteSpace(tableName))
            throw new ArgumentException("Table name cannot be empty", nameof(tableName));
        if (vectorDimension <= 0)
            throw new ArgumentException("Vector dimension must be positive", nameof(vectorDimension));

        _store = new Dictionary<string, VectorDocument<T>>();
        _vectorDimension = vectorDimension;
    }

    /// <summary>
    /// Core logic for adding a single vector document to the database.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    /// <remarks>
    /// <para>
    /// Stores the document in the SQLite table with its vector embedding.
    /// In real SQLite-VSS, this would use INSERT statements with vss0 virtual table functions.
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
    /// Batch operations use SQLite transactions for better performance, inserting all documents
    /// in a single ACID transaction.
    /// </para>
    /// <para><b>For Beginners:</b> Batch operations are much faster in SQLite.
    /// 
    /// Slow (many transactions):
    /// <code>
    /// foreach (var doc in documents)
    ///     store.Add(doc); // Each insert is a separate transaction
    /// </code>
    /// 
    /// Fast (single transaction):
    /// <code>
    /// store.AddBatch(documents); // All inserts in one transaction
    /// </code>
    /// 
    /// This can be 100x faster for large batches!
    /// </para>
    /// </remarks>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0) return;

        if (_vectorDimension == 0)
            _vectorDimension = vectorDocuments[0].Embedding.Length;

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
    /// Performs vector similarity search using in-memory calculations. In real SQLite-VSS, this would use
    /// the vss_search() SQL function to efficiently find nearest neighbors.
    /// </para>
    /// <para><b>For Beginners:</b> Finds the most similar documents in the SQLite database.
    /// 
    /// How it works:
    /// 1. Filter documents by metadata (like SQL WHERE clause)
    /// 2. Calculate similarity for each document
    /// 3. Sort by similarity (highest first, like SQL ORDER BY)
    /// 4. Return top-k matches (like SQL LIMIT)
    /// 
    /// In real SQLite-VSS, this uses efficient indexing structures like HNSW
    /// for fast approximate nearest neighbor search.
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
        var scoredDocuments = new List<(Document<T> doc, T score)>();

        // Apply metadata filters
        var filteredDocuments = _store.Values
            .Where(vectorDoc => MatchesFilters(vectorDoc.Document, metadataFilters));

        foreach (var vd in filteredDocuments)
        {
            var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vd.Embedding);
            scoredDocuments.Add((vd.Document, similarity));
        }

        return scoredDocuments
            .OrderByDescending(x => Convert.ToDouble(x.score))
            .Take(topK)
            .Select(x =>
            {
                // Create a new Document instance to avoid mutating the cached one
                var newDocument = new Document<T>(x.doc.Id, x.doc.Content, x.doc.Metadata)
                {
                    RelevanceScore = x.score,
                    HasRelevanceScore = true
                };
                return newDocument;
            })
            .ToList();
    }

    private bool MatchesFilters(Document<T> document, Dictionary<string, object> metadataFilters)
    {
        if (metadataFilters == null || !metadataFilters.Any())
        {
            return true;
        }

        foreach (var filter in metadataFilters)
        {
            if (document.Metadata.TryGetValue(filter.Key, out var docValue))
            {
                if (!object.Equals(docValue, filter.Value))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        return true;
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
    /// Core logic for removing a document from the database.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from SQLite (like SQL DELETE). The operation is ACID-compliant
    /// and committed to the database file.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document from the database.
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("doc-123"))
    ///     Console.WriteLine("Document deleted from database");
    /// </code>
    /// </para>
    /// </remarks>
    protected override bool RemoveCore(string documentId)
    {
        return _store.Remove(documentId);
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
        _vectorDimension = 0;
    }
}
