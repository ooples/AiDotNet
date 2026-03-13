
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Pinecone-inspired document store with index-based vector organization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation provides an in-memory simulation of Pinecone, a fully managed vector database service.
    /// It organizes documents into named indices and uses cosine similarity for retrieval.
    /// </para>
    /// <para><b>For Beginners:</b> Pinecone is a popular cloud-based vector database.
    /// 
    /// Think of indices like separate databases:
    /// - Each index has a unique name (like "ProductSearchIndex")
    /// - Documents are grouped by use case
    /// - Makes it easy to manage multiple vector search applications
    /// 
    /// This in-memory version is good for:
    /// - Prototyping before using real Pinecone
    /// - Testing Pinecone-style index organization
    /// - Small to medium document collections (< 100K documents)
    /// 
    /// Real Pinecone provides:
    /// - Fully managed cloud service (no infrastructure to manage)
    /// - Auto-scaling to handle any load
    /// - Advanced filtering and hybrid search
    /// - Sub-50ms query latency at scale
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class PineconeDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly string _indexName;
        private int _vectorDimension;

        /// <summary>
        /// Gets the number of documents currently stored in the index.
        /// </summary>
        public override int DocumentCount => _documents.Count;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this index.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Initializes a new instance of the PineconeDocumentStore class.
        /// </summary>
        /// <param name="indexName">The name of the index to organize documents.</param>
        /// <param name="initialCapacity">The initial capacity for the internal dictionary (default: 1000).</param>
        /// <exception cref="ArgumentException">Thrown when index name is empty or initial capacity is not positive.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> Creates a new Pinecone-style index.
        /// 
        /// Example:
        /// <code>
        /// // Create an index for product vectors
        /// var store = new PineconeDocumentStore&lt;float&gt;("ProductIndex");
        /// 
        /// // Create a larger index for articles
        /// var bigStore = new PineconeDocumentStore&lt;double&gt;("ArticleIndex", 10000);
        /// </code>
        /// 
        /// The index name helps organize different vector collections.
        /// </para>
        /// </remarks>
        public PineconeDocumentStore(string indexName, int initialCapacity = 1000)
        {
            if (string.IsNullOrWhiteSpace(indexName))
                throw new ArgumentException("Index name cannot be empty", nameof(indexName));
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _indexName = indexName;
            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _vectorDimension = 0;
        }

        /// <summary>
        /// Core logic for adding a single vector document to the index.
        /// </summary>
        /// <param name="vectorDocument">The validated vector document to add.</param>
        /// <remarks>
        /// <para>
        /// The first document added determines the vector dimension for the entire index.
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
        /// Batch addition is significantly more efficient than adding documents one at a time.
        /// All documents in the batch must have embeddings with the same dimension.
        /// </para>
        /// <para><b>For Beginners:</b> Adding documents in batches is much faster.
        /// 
        /// Bad (slow):
        /// <code>
        /// foreach (var doc in documents)
        ///     store.Add(doc);
        /// </code>
        /// 
        /// Good (fast):
        /// <code>
        /// store.AddBatch(documents);
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
        /// Performs a similarity search across all documents in the index, optionally filtering
        /// by metadata. Results are ordered by decreasing cosine similarity and limited to top-k matches.
        /// </para>
        /// <para><b>For Beginners:</b> Finds the most similar documents to your query.
        /// 
        /// How it works:
        /// 1. Filter documents by metadata (if provided)
        /// 2. Calculate similarity between query and each document's embedding
        /// 3. Sort by similarity (highest first)
        /// 4. Return the top-k best matches
        /// 
        /// Example:
        /// <code>
        /// // Find 10 most similar products
        /// var results = store.GetSimilar(queryVector, topK: 10);
        /// 
        /// // Find similar products in "Electronics" category
        /// var filters = new Dictionary&lt;string, object&gt; { ["category"] = "Electronics" };
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
        /// <para><b>For Beginners:</b> Gets a specific document by its ID.
        /// 
        /// Example:
        /// <code>
        /// var doc = store.GetById("product-123");
        /// if (doc != null)
        ///     Console.WriteLine($"Product: {doc.Content}");
        /// </code>
        /// </para>
        /// </remarks>
        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
        }

        /// <summary>
        /// Core logic for removing a document from the index.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>True if the document was found and removed; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// Removes the document from the index. If this was the last document, the vector dimension
        /// is reset to 0, allowing a new dimension on the next add operation.
        /// </para>
        /// <para><b>For Beginners:</b> Deletes a document from the index.
        /// 
        /// Example:
        /// <code>
        /// bool removed = store.Remove("product-123");
        /// if (removed)
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
        /// Core logic for retrieving all documents in the index.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents in the index in no particular order. Vector embeddings are not included.
        /// For large indices, consider the memory impact of loading all documents at once.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document in the index.
        /// 
        /// Use cases:
        /// - Export all documents for backup
        /// - Migrate to a different index or store
        /// - Bulk analysis or processing
        /// - Debugging to see what's indexed
        /// 
        /// Warning: For large indices (> 10K documents), this can use a lot of memory.
        /// 
        /// Example:
        /// <code>
        /// // Get all documents
        /// var allDocs = store.GetAll().ToList();
        /// Console.WriteLine($"Total in index: {allDocs.Count}");
        /// 
        /// // Export to file
        /// var json = JsonConvert.SerializeObject(allDocs);
        /// File.WriteAllText($"{_indexName}_backup.json", json);
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the index and resets the vector dimension.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears all documents from the index and resets the vector dimension to 0.
        /// The index name remains unchanged and is ready to accept new documents.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties the index.
        /// 
        /// After calling Clear():
        /// - All documents are removed
        /// - Vector dimension resets to 0
        /// - Index name stays the same
        /// - Ready for new documents (even with different dimensions)
        /// 
        /// Use with caution - this operation cannot be undone!
        /// 
        /// Example:
        /// <code>
        /// store.Clear();
        /// Console.WriteLine($"Documents in index: {store.DocumentCount}"); // 0
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

