using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Pinecone-inspired document store with index-based vector organization.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class PineconeDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly string _indexName;
        private int _vectorDimension;

        public override int DocumentCount => _documents.Count;
        public override int VectorDimension => _vectorDimension;

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

        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            if (_documents.Count == 0)
            {
                _vectorDimension = vectorDocument.Embedding.Length;
            }

            _documents[vectorDocument.Document.Id] = vectorDocument;
        }

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

        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _documents.TryGetValue(documentId, out var vectorDoc) ? vectorDoc.Document : null;
        }

        protected override bool RemoveCore(string documentId)
        {
            var removed = _documents.Remove(documentId);
            if (removed && _documents.Count == 0)
            {
                _vectorDimension = 0;
            }
            return removed;
        }

        public override void Clear()
        {
            _documents.Clear();
            _vectorDimension = 0;
        }
    }
}

