using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns
{
    public class ChainOfThoughtRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly int _reasoningSteps;

        public ChainOfThoughtRetriever(IRetriever<T> baseRetriever, int reasoningSteps = 3)
        {
            _baseRetriever = baseRetriever ?? throw new System.ArgumentNullException(nameof(baseRetriever));
            _reasoningSteps = reasoningSteps > 0 ? reasoningSteps : throw new System.ArgumentOutOfRangeException(nameof(reasoningSteps));
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var results = new List<Document<T>>();
            var currentQuery = query;

            for (int step = 0; step < _reasoningSteps; step++)
            {
                var stepResults = _baseRetriever.Retrieve(currentQuery, topK, metadataFilters).ToList();
                
                foreach (var doc in stepResults)
                {
                    if (!results.Any(d => d.Id == doc.Id))
                    {
                        results.Add(doc);
                    }
                }

                if (stepResults.Count > 0)
                {
                    currentQuery = $"{query} Context: {stepResults[0].Content}";
                }
            }

            return results.OrderByDescending(d => Convert.ToDouble(d.RelevanceScore)).Take(topK).ToList();
        }
    }
}

