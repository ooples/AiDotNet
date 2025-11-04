using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns
{
    public class FLARERetriever<T> : RetrieverBase<T> where T : struct
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly int _activeRetrievalSteps;
        
        protected override INumericOperations<T> NumOps { get; }

        public FLARERetriever(IRetriever<T> baseRetriever, int activeRetrievalSteps = 3)
        {
            _baseRetriever = baseRetriever ?? throw new System.ArgumentNullException(nameof(baseRetriever));
            _activeRetrievalSteps = activeRetrievalSteps > 0 ? activeRetrievalSteps : throw new System.ArgumentOutOfRangeException(nameof(activeRetrievalSteps));
            NumOps = NumericOperationsFactory.GetOperations<T>();
        }

        protected override List<Document<T>> RetrieveCore(string query, int topK)
        {
            var allResults = new List<Document<T>>();
            var currentQuery = query;

            for (int step = 0; step < _activeRetrievalSteps; step++)
            {
                var stepResults = _baseRetriever.Retrieve(currentQuery, topK);
                
                foreach (var doc in stepResults)
                {
                    if (!allResults.Any(d => d.Id == doc.Id))
                    {
                        allResults.Add(doc);
                    }
                }

                if (stepResults.Count > 0)
                {
                    var topDoc = stepResults.OrderByDescending(d => NumOps.ToDouble(d.RelevanceScore)).First();
                    currentQuery = $"{query} {topDoc.Content.Substring(0, System.Math.Min(200, topDoc.Content.Length))}";
                }
            }

            return allResults.OrderByDescending(d => NumOps.ToDouble(d.RelevanceScore)).Take(topK).ToList();
        }
    }
}

