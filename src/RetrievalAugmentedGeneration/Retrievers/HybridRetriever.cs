
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Hybrid retriever combining dense and sparse retrieval strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HybridRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _denseRetriever;
        private readonly IRetriever<T> _sparseRetriever;
        private readonly T _denseWeight;
        private readonly T _sparseWeight;

        public HybridRetriever(
            IRetriever<T> denseRetriever,
            IRetriever<T> sparseRetriever,
            double denseWeight = 0.7,
            double sparseWeight = 0.3,
            int defaultTopK = 5)
            : base(defaultTopK)
        {
            if (denseRetriever == null)
                throw new ArgumentNullException(nameof(denseRetriever));
            if (sparseRetriever == null)
                throw new ArgumentNullException(nameof(sparseRetriever));

            _denseRetriever = denseRetriever;
            _sparseRetriever = sparseRetriever;
            _denseWeight = NumOps.FromDouble(denseWeight);
            _sparseWeight = NumOps.FromDouble(sparseWeight);
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var denseResults = _denseRetriever.Retrieve(query, topK * 2, metadataFilters);
            var sparseResults = _sparseRetriever.Retrieve(query, topK * 2, metadataFilters);

            var combinedScores = new Dictionary<string, T>();

            foreach (var doc in denseResults.Where(d => d.HasRelevanceScore))
            {
                var score = NumOps.Multiply(_denseWeight, doc.RelevanceScore);
                combinedScores[doc.Id] = score;
            }

            foreach (var doc in sparseResults.Where(doc => doc.HasRelevanceScore))
            {
                var sparseScore = NumOps.Multiply(_sparseWeight, doc.RelevanceScore);
                combinedScores[doc.Id] = combinedScores.TryGetValue(doc.Id, out var existingScore)
                    ? NumOps.Add(existingScore, sparseScore)
                    : sparseScore;
            }

            var allDocuments = denseResults.Concat(sparseResults)
                .GroupBy(d => d.Id)
                .Select(g => g.First())
                .ToList();

            var results = allDocuments
                .Where(doc => combinedScores.TryGetValue(doc.Id, out _))
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
    }
}
