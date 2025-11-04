using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// FAISS-inspired document store with indexed vectors for efficient similarity search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class FAISSDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents;
        private readonly Dictionary<int, Vector<T>> _indexedVectors;
        private int _vectorDimension;
        private int _currentIndex;

        public override int DocumentCount => _documents.Count;
        public override int VectorDimension => _vectorDimension;

        public FAISSDocumentStore(int initialCapacity = 1000)
        {
            if (initialCapacity <= 0)
                throw new ArgumentException("Initial capacity must be greater than zero", nameof(initialCapacity));

            _documents = new Dictionary<string, VectorDocument<T>>(initialCapacity);
            _indexedVectors = new Dictionary<int, Vector<T>>(initialCapacity);
            _vectorDimension = 0;
            _currentIndex = 0;
        }

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
                _currentIndex = 0;
                _indexedVectors.Clear();
            }
            return removed;
        }

        public override void Clear()
        {
            _documents.Clear();
            _indexedVectors.Clear();
            _vectorDimension = 0;
            _currentIndex = 0;
        }
    }
}

