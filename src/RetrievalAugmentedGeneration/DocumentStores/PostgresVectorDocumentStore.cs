
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// PostgreSQL with pgvector-inspired document store for relational database vector storage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation provides an in-memory simulation of PostgreSQL with the pgvector extension,
    /// which adds vector similarity search capabilities to the popular relational database.
    /// It organizes documents in table-like structures with cosine similarity for retrieval.
    /// </para>
    /// <para><b>For Beginners:</b> PostgreSQL is a powerful open-source relational database, and pgvector adds AI capabilities.
    /// 
    /// Think of it like a smart database table:
    /// - Each table stores documents with vector embeddings
    /// - Combines traditional SQL queries with vector search
    /// - Leverage existing PostgreSQL infrastructure
    /// 
    /// This in-memory version is good for:
    /// - Prototyping pgvector applications
    /// - Testing table-based organization
    /// - Small to medium collections (< 100K documents)
    /// 
    /// Real PostgreSQL + pgvector provides:
    /// - ACID transactions for data integrity
    /// - Complex SQL joins with vector search
    /// - Proven reliability and scalability
    /// - Integration with existing database infrastructure
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class PostgresVectorDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly string _tableName;
        private int _vectorDimension;

        /// <summary>
        /// Gets the number of documents currently stored in the table.
        /// </summary>
        public override int DocumentCount => _documents.Count;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this table.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Initializes a new instance of the PostgresVectorDocumentStore class.
        /// </summary>
        /// <param name="tableName">The table name to organize documents.</param>
        /// <param name="initialCapacity">The initial capacity for the internal dictionary (default: 1000).</param>
        /// <exception cref="ArgumentException">Thrown when table name is empty or initial capacity is not positive.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> Creates a new pgvector-style document table.
        /// 
        /// Example:
        /// <code>
        /// // Create a table for documents
        /// var store = new PostgresVectorDocumentStore&lt;float&gt;("documents");
        /// 
        /// // Create a table for embeddings
        /// var embStore = new PostgresVectorDocumentStore&lt;double&gt;("embeddings", 10000);
        /// </code>
        /// 
        /// The table name helps organize different document collections.
        /// </para>
        /// </remarks>
        public PostgresVectorDocumentStore(string tableName, int initialCapacity = 1000)
        {
            if (string.IsNullOrWhiteSpace(tableName))
                throw new ArgumentException("Table name cannot be empty", nameof(tableName));
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _tableName = tableName;
            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _vectorDimension = 0;
        }

        /// <summary>
        /// Core logic for adding a single vector document to the table.
        /// </summary>
        /// <param name="vectorDocument">The validated vector document to add.</param>
        /// <remarks>
        /// <para>
        /// The first document added determines the vector dimension for all documents in this table.
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
        /// Batch operations are more efficient than adding documents individually, similar to PostgreSQL's
        /// bulk insert capabilities. All documents must have embeddings with the same dimension.
        /// </para>
        /// <para><b>For Beginners:</b> Adding many documents at once is much faster.
        /// 
        /// Slow way:
        /// <code>
        /// foreach (var doc in documents)
        ///     store.Add(doc); // Many individual inserts
        /// </code>
        /// 
        /// Fast way:
        /// <code>
        /// store.AddBatch(documents); // Single bulk insert
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
        /// Performs vector similarity search across all documents in the table, optionally filtering by metadata.
        /// In real pgvector, this would use the vector similarity operators (<->, <=>, <#>) in SQL.
        /// </para>
        /// <para><b>For Beginners:</b> Finds the most similar documents to your query.
        /// 
        /// How it works:
        /// 1. Filter documents by metadata (like SQL WHERE clause)
        /// 2. Calculate similarity between query and each document
        /// 3. Sort by similarity (highest first, like SQL ORDER BY)
        /// 4. Return top-k matches (like SQL LIMIT)
        /// 
        /// Example:
        /// <code>
        /// // Find 10 most similar documents
        /// var results = store.GetSimilar(queryVector, topK: 10);
        /// 
        /// // Find similar documents from 2024
        /// var filters = new Dictionary&lt;string, object&gt; { ["year"] = "2024" };
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
                    // Create a new Document instance to avoid mutating the cached one
                    var newDocument = new Document<T>(x.Document.Id, x.Document.Content, x.Document.Metadata)
                    {
                        RelevanceScore = x.Score,
                        HasRelevanceScore = true
                    };
                    return newDocument;
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
            return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
        }

        /// <summary>
        /// Core logic for removing a document from the table.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>True if the document was found and removed; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// Removes the document from the table (like SQL DELETE). If this was the last document,
        /// the vector dimension is reset to 0, allowing a new dimension on next add.
        /// </para>
        /// <para><b>For Beginners:</b> Deletes a document from the table.
        /// 
        /// Example:
        /// <code>
        /// if (store.Remove("doc-123"))
        ///     Console.WriteLine("Document deleted");
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
        /// Core logic for retrieving all documents in the table.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents in the table in no particular order (like SQL SELECT * FROM table).
        /// Vector embeddings are not included in the results.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document in the table.
        /// 
        /// Use cases:
        /// - Export all documents for backup
        /// - Migrate to a different table or database
        /// - Bulk processing or analysis
        /// - Debugging to see all stored data
        /// 
        /// Warning: For large tables (> 10K documents), this can use significant memory.
        /// In production, consider using pagination with LIMIT and OFFSET.
        /// 
        /// Example:
        /// <code>
        /// // Get all documents
        /// var allDocs = store.GetAll().ToList();
        /// Console.WriteLine($"Total rows in table: {allDocs.Count}");
        /// 
        /// // Export to JSON
        /// var json = JsonConvert.SerializeObject(allDocs);
        /// File.WriteAllText($"{_tableName}_export.json", json);
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the table and resets the vector dimension.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears all documents from the table and resets the vector dimension to 0 (like SQL TRUNCATE TABLE).
        /// The table name remains unchanged and is ready to accept new documents.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties the table.
        /// 
        /// After calling Clear():
        /// - All documents are removed (like DELETE FROM table)
        /// - Vector dimension resets to 0
        /// - Table name stays the same
        /// - Ready for new documents (even with different dimensions)
        /// 
        /// Use with caution - this cannot be undone!
        /// 
        /// Example:
        /// <code>
        /// store.Clear();
        /// Console.WriteLine($"Rows in table: {store.DocumentCount}"); // 0
        /// </code>
        /// </para>
        /// </remarks>
        public override void Clear()
        {
            _documents.Clear();
            _vectorDimension = 0;
        }
    }
}

