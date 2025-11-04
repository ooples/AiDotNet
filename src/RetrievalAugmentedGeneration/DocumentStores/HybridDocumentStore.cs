using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Hybrid document store combining vector and keyword search strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HybridDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly IDocumentStore<T> _vectorStore;
        private readonly IDocumentStore<T> _keywordStore;
        private readonly T _vectorWeight;
        private readonly T _keywordWeight;

        public override int DocumentCount => _vectorStore.DocumentCount;
        public override int VectorDimension => _vectorStore.VectorDimension;

        public HybridDocumentStore(
            IDocumentStore<T> vectorStore,
            IDocumentStore<T> keywordStore,
            T vectorWeight,
            T keywordWeight)
        {
            if (vectorStore == null)
                throw new ArgumentNullException(nameof(vectorStore));
            if (keywordStore == null)
                throw new ArgumentNullException(nameof(keywordStore));

            _vectorStore = vectorStore;
            _keywordStore = keywordStore;
            _vectorWeight = vectorWeight;
            _keywordWeight = keywordWeight;
        }

        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            _vectorStore.Add(vectorDocument);
            _keywordStore.Add(vectorDocument);
        }

        protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
        {
            _vectorStore.AddBatch(vectorDocuments);
            _keywordStore.AddBatch(vectorDocuments);
        }

        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            var vectorResults = _vectorStore.GetSimilarWithFilters(queryVector, topK * 2, metadataFilters);
            var keywordResults = _keywordStore.GetSimilarWithFilters(queryVector, topK * 2, metadataFilters);

            var combinedScores = new Dictionary<string, T>();

            foreach (var doc in vectorResults.Where(d => d.HasRelevanceScore))
            {
                var score = NumOps.Multiply(_vectorWeight, doc.RelevanceScore);
                combinedScores[doc.Id] = score;
            }

            foreach (var doc in keywordResults.Where(doc => doc.HasRelevanceScore))
            {
                var keywordScore = NumOps.Multiply(_keywordWeight, doc.RelevanceScore);
                if (combinedScores.TryGetValue(doc.Id, out var existingScore))
                {
                    combinedScores[doc.Id] = NumOps.Add(existingScore, keywordScore);
                }
                else
                {
                    combinedScores[doc.Id] = keywordScore;
                }
            }

            var allDocuments = vectorResults.Concat(keywordResults)
                .GroupBy(d => d.Id)
                .Select(g => g.First())
                .ToList();

            var results = allDocuments
                .Where(doc => combinedScores.ContainsKey(doc.Id))
                .OrderByDescending(doc => combinedScores[doc.Id])
                .Take(topK)
                .Select(doc =>
                {
                    doc.RelevanceScore = combinedScores[doc.Id];
                    doc.HasRelevanceScore = true;
                    return doc;
                })
                .ToList();

            return results;
        }

        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _vectorStore.GetById(documentId);
        }

        protected override bool RemoveCore(string documentId)
        {
            var removed1 = _vectorStore.Remove(documentId);
            var removed2 = _keywordStore.Remove(documentId);
            return removed1 || removed2;
        }

        public override void Clear()
        {
            _vectorStore.Clear();
            _keywordStore.Clear();
        }
    }
}

