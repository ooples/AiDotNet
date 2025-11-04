using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns
{
    public class SelfCorrectingRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly int _maxCorrectionAttempts;
        private readonly T _relevanceThreshold;

        public SelfCorrectingRetriever(IRetriever<T> baseRetriever, T relevanceThreshold, int maxCorrectionAttempts = 3)
        {
            _baseRetriever = baseRetriever ?? throw new System.ArgumentNullException(nameof(baseRetriever));
            _relevanceThreshold = relevanceThreshold;
            _maxCorrectionAttempts = maxCorrectionAttempts > 0 ? maxCorrectionAttempts : throw new System.ArgumentOutOfRangeException(nameof(maxCorrectionAttempts));
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var results = _baseRetriever.Retrieve(query, topK, metadataFilters).ToList();
            var attempts = 0;

            while (attempts < _maxCorrectionAttempts && NeedsCorrection(results))
            {
                var refinedQuery = RefineQuery(query, results);
                results = _baseRetriever.Retrieve(refinedQuery, topK, metadataFilters).ToList();
                attempts++;
            }

            return results;
        }

        private bool NeedsCorrection(List<Document<T>> documents)
        {
            if (documents == null || documents.Count == 0)
            {
                return true;
            }

            var avgScore = NumOps.Divide(
                documents.Aggregate(NumOps.Zero, (acc, doc) => NumOps.Add(acc, doc.RelevanceScore)),
                NumOps.FromDouble(documents.Count)
            );

            return NumOps.LessThan(avgScore, _relevanceThreshold);
        }

        private string RefineQuery(string originalQuery, List<Document<T>> previousResults)
        {
            if (previousResults == null || previousResults.Count == 0)
            {
                return originalQuery;
            }

            var topDoc = previousResults.OrderByDescending(d => Convert.ToDouble(d.RelevanceScore)).First();
            var snippet = topDoc.Content.Substring(0, System.Math.Min(100, topDoc.Content.Length));
            
            return $"{originalQuery} {snippet}";
        }
    }
}

