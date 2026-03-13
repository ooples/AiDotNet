
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Qdrant-inspired document store with collection-based organization and payload filtering.
    /// Provides in-memory simulation of Qdrant features including collection management and efficient filtering.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class QdrantDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly string _collectionName;
        private int _vectorDimension;
        private readonly Dictionary<string, HashSet<string>> _payloadIndex;

        public override int DocumentCount => _documents.Count;
        public override int VectorDimension => _vectorDimension;

        public string CollectionName { get; private set; }

        public QdrantDocumentStore(string collectionName, int initialCapacity = 1000)
        {
            if (string.IsNullOrWhiteSpace(collectionName))
                throw new ArgumentException("Collection name cannot be empty", nameof(collectionName));
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _collectionName = collectionName;
            CollectionName = collectionName;
            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _payloadIndex = new Dictionary<string, HashSet<string>>();
            _vectorDimension = 0;
        }

        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            if (_documents.Count == 0)
            {
                _vectorDimension = vectorDocument.Embedding.Length;
            }

            _documents[vectorDocument.Document.Id] = vectorDocument;
            IndexPayload(vectorDocument.Document);
        }

        protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
        {
            if (vectorDocuments.Count == 0)
                return;

            if (_documents.Count == 0)
            {
                _vectorDimension = vectorDocuments[0].Embedding.Length;
            }

            foreach (var vectorDoc in vectorDocuments)
            {
                _documents[vectorDoc.Document.Id] = vectorDoc;
                IndexPayload(vectorDoc.Document);
            }
        }

        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            var scoredDocuments = new List<(Document<T> Document, T Score)>();

            var candidateIds = GetFilteredCandidates(metadataFilters);
            IEnumerable<VectorDocument<T>> candidates;

            if (candidateIds != null)
            {
                candidates = candidateIds
                    .Where(id => _documents.ContainsKey(id))
                    .Select(id => _documents[id]);
            }
            else
            {
                candidates = _documents.Values;
            }

            var matchingDocuments = candidates
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

        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
        }

        protected override bool RemoveCore(string documentId)
        {
            if (!_documents.TryGetValue(documentId, out var vectorDoc))
                return false;

            RemoveFromPayloadIndex(vectorDoc.Document);
            _documents.Remove(documentId);

            if (_documents.Count == 0)
            {
                _vectorDimension = 0;
            }

            return true;
        }

        /// <summary>
        /// Core logic for retrieving all documents in the collection.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents from the Qdrant collection in no particular order.
        /// Vector embeddings are not included in the results, only document content and metadata.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document in the collection.
        /// 
        /// Use cases:
        /// - Export all documents for backup
        /// - Migrate to a different collection
        /// - Bulk processing or analysis
        /// - Debugging payload indices
        /// 
        /// Warning: For large collections (> 10K documents), this can use significant memory.
        /// In real Qdrant, consider using scroll API with pagination for large collections.
        /// 
        /// Example:
        /// <code>
        /// // Get all documents
        /// var allDocs = store.GetAll().ToList();
        /// Console.WriteLine($"Total documents in {_collectionName}: {allDocs.Count}");
        /// 
        /// // Export to JSON
        /// var json = JsonConvert.SerializeObject(allDocs);
        /// File.WriteAllText($"{_collectionName}_export.json", json);
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the collection and clears all indices.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears all documents, payload indices, and resets the vector dimension to 0.
        /// The collection name remains unchanged and is ready to accept new documents.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties the collection and all its indices.
        /// 
        /// After calling Clear():
        /// - All documents are removed
        /// - Payload index is cleared
        /// - Vector dimension resets to 0
        /// - Collection name stays the same
        /// - Ready for new documents
        /// 
        /// Use with caution - this cannot be undone!
        /// 
        /// Example:
        /// <code>
        /// store.Clear();
        /// Console.WriteLine($"Documents in {CollectionName}: {store.DocumentCount}"); // 0
        /// </code>
        /// </para>
        /// </remarks>
        public override void Clear()
        {
            _documents.Clear();
            _payloadIndex.Clear();
            _vectorDimension = 0;
        }

        private void IndexPayload(Document<T> document)
        {
            foreach (var kvp in document.Metadata)
            {
                var payloadKey = CreatePayloadKey(kvp.Key, kvp.Value);

                if (!_payloadIndex.ContainsKey(payloadKey))
                {
                    _payloadIndex[payloadKey] = new HashSet<string>();
                }

                _payloadIndex[payloadKey].Add(document.Id);
            }
        }

        private void RemoveFromPayloadIndex(Document<T> document)
        {
            foreach (var kvp in document.Metadata)
            {
                var payloadKey = CreatePayloadKey(kvp.Key, kvp.Value);

                if (_payloadIndex.TryGetValue(payloadKey, out var docIds))
                {
                    docIds.Remove(document.Id);
                    if (docIds.Count == 0)
                    {
                        _payloadIndex.Remove(payloadKey);
                    }
                }
            }
        }

        private HashSet<string>? GetFilteredCandidates(Dictionary<string, object> metadataFilters)
        {
            if (metadataFilters.Count == 0)
                return null;

            HashSet<string>? candidateIds = null;

            foreach (var filter in metadataFilters)
            {
                var payloadKey = CreatePayloadKey(filter.Key, filter.Value);

                if (_payloadIndex.TryGetValue(payloadKey, out var docIds))
                {
                    if (candidateIds == null)
                    {
                        candidateIds = new HashSet<string>(docIds);
                    }
                    else
                    {
                        candidateIds.IntersectWith(docIds);
                    }
                }
                else
                {
                    return new HashSet<string>();
                }
            }

            return candidateIds;
        }

        private static string CreatePayloadKey(string fieldName, object fieldValue)
        {
            var valueStr = fieldValue?.ToString() ?? string.Empty;
            return $"{fieldName}:{valueStr}";
        }
    }
}
