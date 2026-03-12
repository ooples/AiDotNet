
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// FAISS-inspired document store with indexed vectors for efficient similarity search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation provides an in-memory simulation of Facebook AI Similarity Search (FAISS),
    /// using integer-based indexing for fast vector lookup. It maintains both document storage and
    /// a separate index mapping for optimized retrieval operations.
    /// </para>
    /// <para><b>For Beginners:</b> FAISS is Facebook's ultra-fast vector search library.
    /// 
    /// Think of it like a specialized phone book for vectors:
    /// - Each document gets a unique number (index)
    /// - Vectors are stored separately for faster search
    /// - Can handle millions of vectors efficiently
    /// 
    /// This in-memory version is good for:
    /// - Testing FAISS-style indexing patterns
    /// - Medium-sized collections (< 100K documents)
    /// - Prototyping before deploying real FAISS
    /// 
    /// Real FAISS provides:
    /// - GPU acceleration for billion-scale search
    /// - Advanced indexing (IVF, HNSW, Product Quantization)
    /// - Sub-millisecond search on huge datasets
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class FAISSDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly Dictionary<int, Vector<T>> _indexedVectors;
        private int _vectorDimension;
        private int _currentIndex;

        /// <summary>
        /// Gets the number of documents currently stored.
        /// </summary>
        public override int DocumentCount => _documents.Count;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this index.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Initializes a new instance of the FAISSDocumentStore class.
        /// </summary>
        /// <param name="initialCapacity">The initial capacity for internal dictionaries (default: 1000).</param>
        /// <exception cref="ArgumentException">Thrown when initial capacity is not positive.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> Creates a new FAISS-style document store.
        /// 
        /// Example:
        /// <code>
        /// // Create a store for embeddings
        /// var store = new FAISSDocumentStore&lt;float&gt;();
        /// 
        /// // Create a larger store
        /// var bigStore = new FAISSDocumentStore&lt;double&gt;(initialCapacity: 50000);
        /// </code>
        /// 
        /// The initial capacity is just a hint - the store can grow beyond it.
        /// </para>
        /// </remarks>
        public FAISSDocumentStore(int initialCapacity = 1000)
        {
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _indexedVectors = new Dictionary<int, Vector<T>>(initialCapacity);
            _vectorDimension = 0;
            _currentIndex = 0;
        }

        /// <summary>
        /// Core logic for adding a single vector document with automatic indexing.
        /// </summary>
        /// <param name="vectorDocument">The validated vector document to add.</param>
        /// <remarks>
        /// <para>
        /// Assigns a unique integer index to each document for fast lookup in the vector index.
        /// The first document added determines the vector dimension for all subsequent additions.
        /// </para>
        /// </remarks>
        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            if (_documents.Count == 0)
            {
                _vectorDimension = vectorDocument.Embedding.Length;
            }

            var index = _currentIndex++;
            _documents[vectorDocument.Document.Id] = vectorDocument;
            _indexedVectors[index] = vectorDocument.Embedding;
        }

        /// <summary>
        /// Core logic for adding multiple vector documents in a batch with automatic indexing.
        /// </summary>
        /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
        /// <exception cref="ArgumentException">Thrown when a document's embedding has inconsistent dimensions.</exception>
        /// <remarks>
        /// <para>
        /// Batch operations are more efficient than adding documents individually. Each document
        /// receives a sequential integer index for fast vector lookup.
        /// </para>
        /// <para><b>For Beginners:</b> Adding many documents at once is faster.
        /// 
        /// Instead of:
        /// <code>
        /// // Slow - one at a time
        /// foreach (var doc in docs)
        ///     store.Add(doc);
        /// </code>
        /// 
        /// Do this:
        /// <code>
        /// // Fast - all at once
        /// store.AddBatch(docs);
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
                        $"Vector dimension mismatch in batch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length}",
                        nameof(vectorDocuments));

                var index = _currentIndex++;
                _documents[vectorDocument.Document.Id] = vectorDocument;
                _indexedVectors[index] = vectorDocument.Embedding;
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
        /// Searches through all documents, filters by metadata, computes cosine similarity scores,
        /// and returns the top-k most similar documents ordered by descending similarity.
        /// </para>
        /// <para><b>For Beginners:</b> Finds documents most similar to your query.
        /// 
        /// Process:
        /// 1. Filter documents by metadata (if filters provided)
        /// 2. Calculate similarity between query and each document
        /// 3. Sort by similarity score (highest first)
        /// 4. Return top-k matches
        /// 
        /// Example:
        /// <code>
        /// // Find 5 most similar documents
        /// var results = store.GetSimilar(queryVector, topK: 5);
        /// 
        /// // Find 10 similar documents from 2024
        /// var filters = new Dictionary&lt;string, object&gt; { ["year"] = "2024" };
        /// var filtered = store.GetSimilarWithFilters(queryVector, 10, filters);
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
        /// <para><b>For Beginners:</b> Gets a specific document by its ID.
        /// 
        /// Example:
        /// <code>
        /// var doc = store.GetById("doc123");
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
        /// Core logic for removing a document from the store.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>True if the document was found and removed; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// Removes the document from storage. When the last document is removed, the vector dimension
        /// is reset and indices are cleared, allowing a new dimension on next add.
        /// </para>
        /// <para><b>For Beginners:</b> Removes a document from the store.
        /// 
        /// Example:
        /// <code>
        /// if (store.Remove("doc123"))
        ///     Console.WriteLine("Document removed");
        /// else
        ///     Console.WriteLine("Document not found");
        /// </code>
        /// </para>
        /// </remarks>
        protected override bool RemoveCore(string documentId)
        {
            var removed = _documents.Remove(documentId);
            if (removed)
            {
                if (_documents.Count == 0)
                {
                    _vectorDimension = 0;
                    _currentIndex = 0;
                    _indexedVectors.Clear();
                }
                else
                {
                    // Rebuild index from remaining documents to keep FAISS index in sync
                    _indexedVectors.Clear();
                    _currentIndex = 0;
                    foreach (var kvp in _documents)
                    {
                        var index = _currentIndex++;
                        _indexedVectors[index] = kvp.Value.Embedding;
                    }
                }
            }
            return removed;
        }

        /// <summary>
        /// Core logic for retrieving all documents in the store.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents in the store in no particular order. The vector embeddings are not
        /// included in the results - only the document metadata and content.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document in the store.
        /// 
        /// Use cases:
        /// - Export all documents to a file
        /// - Migrate to a different document store
        /// - Bulk processing or analysis
        /// - Debugging - see what's stored
        /// 
        /// Warning: For large collections (> 10K documents), this can use significant memory.
        /// Consider pagination for production systems with large datasets.
        /// 
        /// Example:
        /// <code>
        /// // Get all documents
        /// var allDocs = store.GetAll();
        /// Console.WriteLine($"Total documents: {allDocs.Count()}");
        /// 
        /// // Export to JSON
        /// var json = JsonConvert.SerializeObject(allDocs);
        /// File.WriteAllText("backup.json", json);
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the store and resets all indices.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears all documents, resets the vector dimension to 0, resets the index counter,
        /// and clears the vector index. The store is returned to its initial empty state.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties the store.
        /// 
        /// After calling Clear():
        /// - All documents are gone
        /// - Vector dimension resets to 0
        /// - Index counter resets to 0
        /// - Ready to accept new documents
        /// 
        /// Use with caution - this cannot be undone!
        /// 
        /// Example:
        /// <code>
        /// store.Clear();
        /// Console.WriteLine($"Documents remaining: {store.DocumentCount}"); // 0
        /// </code>
        /// </para>
        /// </remarks>
        public override void Clear()
        {
            _documents.Clear();
            _indexedVectors.Clear();
            _vectorDimension = 0;
            _currentIndex = 0;
        }
    }
}

