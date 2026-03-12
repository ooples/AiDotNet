
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Milvus-inspired document store with collection-based vector organization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation provides a simplified in-memory version inspired by Milvus,
    /// a cloud-native vector database. It organizes documents into named collections
    /// and uses cosine similarity for retrieval operations.
    /// </para>
    /// <para><b>For Beginners:</b> This document store mimics Milvus, a popular cloud-based vector database.
    ///
    /// Think of collections like folders on your computer:
    /// - Each collection has a unique name (like "ResearchPapers" or "CustomerReviews")
    /// - Documents are grouped by topic or purpose
    /// - Makes it easy to organize different types of content separately
    ///
    /// Good for:
    /// - Prototyping before using real Milvus
    /// - Testing Milvus-style organization patterns
    /// - Small to medium document sets (< 100K documents)
    ///
    /// For production use real Milvus which provides:
    /// - Distributed storage across multiple servers
    /// - Advanced indexing (IVF, HNSW) for billion-scale vectors
    /// - GPU acceleration for ultra-fast similarity search
    /// - Persistence and fault tolerance
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class MilvusDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly string _collectionName;
        private int _vectorDimension;

        /// <summary>
        /// Gets the number of documents currently stored in this collection.
        /// </summary>
        public override int DocumentCount => _documents.Count;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this collection.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Initializes a new instance of the MilvusDocumentStore class.
        /// </summary>
        /// <param name="collectionName">The name of the collection to organize documents.</param>
        /// <param name="initialCapacity">The initial capacity for the internal dictionary (default: 1000).</param>
        /// <exception cref="ArgumentException">Thrown when collection name is empty or initial capacity is not positive.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> Creates a new document collection with a specific name.
        ///
        /// Example:
        /// <code>
        /// // Create a collection for research papers
        /// var store = new MilvusDocumentStore&lt;float&gt;("ResearchPapers");
        ///
        /// // Create a larger collection
        /// var bigStore = new MilvusDocumentStore&lt;double&gt;("Articles", initialCapacity: 10000);
        /// </code>
        ///
        /// The initial capacity is a performance hint - the collection can grow beyond it.
        /// </para>
        /// </remarks>
        public MilvusDocumentStore(string collectionName, int initialCapacity = 1000)
        {
            if (string.IsNullOrWhiteSpace(collectionName))
                throw new ArgumentException("Collection name cannot be empty", nameof(collectionName));
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _collectionName = collectionName;
            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _vectorDimension = 0;
        }

        /// <summary>
        /// Core logic for adding a single vector document to the collection.
        /// </summary>
        /// <param name="vectorDocument">The validated vector document to add.</param>
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
        /// Core logic for similarity search with optional metadata filtering.
        /// </summary>
        /// <param name="queryVector">The validated query vector.</param>
        /// <param name="topK">The validated number of documents to return.</param>
        /// <param name="metadataFilters">The validated metadata filters.</param>
        /// <returns>Top-k similar documents ordered by cosine similarity score.</returns>
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
        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
        }

        /// <summary>
        /// Core logic for removing a document from the collection.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>True if the document was found and removed; otherwise, false.</returns>
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
        /// Core logic for retrieving all documents in the collection.
        /// </summary>
        /// <returns>An enumerable of all documents in the collection.</returns>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the collection and resets vector dimension.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> This empties the entire collection.
        ///
        /// After calling Clear():
        /// - All documents are removed
        /// - Vector dimension is reset to 0
        /// - Collection name stays the same
        /// - Ready to accept new documents (even with different dimensions)
        ///
        /// Use with caution - this operation cannot be undone!
        /// </para>
        /// </remarks>
        public override void Clear()
        {
            _documents.Clear();
            _vectorDimension = 0;
        }
    }
}

