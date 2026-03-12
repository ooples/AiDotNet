
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Azure Cognitive Search-inspired document store with field-based indexing and search capabilities.
    /// Provides in-memory simulation of Azure Search features including field-level search and faceted filtering.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class AzureSearchDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly Dictionary<string, Dictionary<string, HashSet<string>>> _invertedIndex;
        private readonly string _serviceName;
        private readonly string _indexName;
        private int _vectorDimension;

        public override int DocumentCount => _documents.Count;
        public override int VectorDimension => _vectorDimension;

        public AzureSearchDocumentStore(string serviceName, string indexName, int initialCapacity = 1000)
        {
            if (string.IsNullOrWhiteSpace(serviceName))
                throw new ArgumentException("Service name cannot be empty", nameof(serviceName));
            if (string.IsNullOrWhiteSpace(indexName))
                throw new ArgumentException("Index name cannot be empty", nameof(indexName));
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _serviceName = serviceName;
            _indexName = indexName;
            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _invertedIndex = new Dictionary<string, Dictionary<string, HashSet<string>>>();
            _vectorDimension = 0;
        }

        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            if (_documents.Count == 0)
            {
                _vectorDimension = vectorDocument.Embedding.Length;
            }

            _documents[vectorDocument.Document.Id] = vectorDocument;
            IndexMetadata(vectorDocument.Document);
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
                IndexMetadata(vectorDoc.Document);
            }
        }

        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            var candidateIds = GetCandidateIds(metadataFilters);
            var scoredDocuments = new List<(Document<T> Document, T Score)>();

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

            RemoveFromIndex(vectorDoc.Document);
            _documents.Remove(documentId);

            if (_documents.Count == 0)
            {
                _vectorDimension = 0;
            }

            return true;
        }

        /// <summary>
        /// Core logic for retrieving all documents in the index.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents from the Azure Search index in no particular order.
        /// Vector embeddings are not included, only document content and metadata.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document in the index.
        /// 
        /// Use cases:
        /// - Export all documents for backup
        /// - Migrate to a different index or service
        /// - Bulk reindexing or analysis
        /// - Debugging facet indices
        /// 
        /// Warning: For large indices (> 10K documents), this can use significant memory.
        /// In real Azure Search, use continuation tokens for pagination.
        /// 
        /// Example:
        /// <code>
        /// // Get all documents
        /// var allDocs = store.GetAll().ToList();
        /// Console.WriteLine($"Total documents in {_indexName}: {allDocs.Count}");
        /// 
        /// // Export to JSON
        /// var json = JsonConvert.SerializeObject(allDocs);
        /// File.WriteAllText($"{_serviceName}_{_indexName}_export.json", json);
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _documents.Values.Select(vd => vd.Document).ToList();
        }

        /// <summary>
        /// Removes all documents from the index and clears all inverted indices.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears all documents, field-level inverted indices, and resets the vector dimension to 0.
        /// The service and index names remain unchanged and the index is ready to accept new documents.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties the Azure Search index and all its facet indices.
        /// 
        /// After calling Clear():
        /// - All documents are removed
        /// - Inverted index is cleared (all facets)
        /// - Vector dimension resets to 0
        /// - Index is ready for new documents
        /// 
        /// Use with caution - this cannot be undone!
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
            _invertedIndex.Clear();
            _vectorDimension = 0;
        }

        private void IndexMetadata(Document<T> document)
        {
            foreach (var kvp in document.Metadata)
            {
                var fieldName = kvp.Key;
                var fieldValue = kvp.Value;

                if (!_invertedIndex.ContainsKey(fieldName))
                {
                    _invertedIndex[fieldName] = new Dictionary<string, HashSet<string>>();
                }

                // Create normalized key that preserves type information for accurate comparisons
                var indexKey = NormalizeMetadataValue(fieldValue);

                if (!_invertedIndex[fieldName].ContainsKey(indexKey))
                {
                    _invertedIndex[fieldName][indexKey] = new HashSet<string>();
                }

                _invertedIndex[fieldName][indexKey].Add(document.Id);
            }
        }

        private string NormalizeMetadataValue(object? value)
        {
            if (value == null)
                return "null";

            // Preserve type information in the key to ensure accurate comparisons
            return value switch
            {
                bool b => $"bool:{(b ? "true" : "false")}",
                int i => $"int:{i}",
                long l => $"long:{l}",
                float f => $"float:{f:R}",
                double d => $"double:{d:R}",
                decimal dec => $"decimal:{dec}",
                _ => $"string:{value}"
            };
        }

        private void RemoveFromIndex(Document<T> document)
        {
            foreach (var kvp in document.Metadata)
            {
                var fieldName = kvp.Key;
                var fieldValue = NormalizeMetadataValue(kvp.Value);

                if (_invertedIndex.TryGetValue(fieldName, out var fieldIndex))
                {
                    if (fieldIndex.TryGetValue(fieldValue, out var docIds))
                    {
                        docIds.Remove(document.Id);
                        if (docIds.Count == 0)
                        {
                            fieldIndex.Remove(fieldValue);
                        }
                    }
                    if (fieldIndex.Count == 0)
                    {
                        _invertedIndex.Remove(fieldName);
                    }
                }
            }
        }

        private HashSet<string>? GetCandidateIds(Dictionary<string, object> metadataFilters)
        {
            if (metadataFilters.Count == 0)
                return null;

            HashSet<string>? candidateIds = null;

            foreach (var filter in metadataFilters)
            {
                var fieldName = filter.Key;
                var indexKey = NormalizeMetadataValue(filter.Value);

                if (_invertedIndex.TryGetValue(fieldName, out var fieldIndex))
                {
                    if (fieldIndex.TryGetValue(indexKey, out var docIds))
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
                else
                {
                    return new HashSet<string>();
                }
            }

            return candidateIds;
        }
    }
}
