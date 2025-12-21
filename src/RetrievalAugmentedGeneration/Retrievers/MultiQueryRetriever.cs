
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Multi-query retriever that generates multiple query variations and merges results.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class MultiQueryRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly int _numQueries;

        public MultiQueryRetriever(IRetriever<T> baseRetriever, int numQueries = 3, int defaultTopK = 5) : base(defaultTopK)
        {
            if (baseRetriever == null)
                throw new ArgumentNullException(nameof(baseRetriever));
            if (numQueries <= 0)
                throw new ArgumentException("Number of queries must be positive", nameof(numQueries));

            _baseRetriever = baseRetriever;
            _numQueries = numQueries;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queries = GenerateQueries(query);
            var allResults = new Dictionary<string, (Document<T> doc, T score)>();

            foreach (var q in queries)
            {
                var results = _baseRetriever.Retrieve(q, topK, metadataFilters);

                foreach (var doc in results.Where(d => d.HasRelevanceScore))
                {
                    if (allResults.TryGetValue(doc.Id, out var existing))
                    {
                        var newScore = NumOps.Add(existing.score, doc.RelevanceScore);
                        allResults[doc.Id] = (doc, newScore);
                    }
                    else
                    {
                        allResults[doc.Id] = (doc, doc.RelevanceScore);
                    }
                }
            }

            var merged = allResults.Values
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    x.doc.HasRelevanceScore = true;
                    return x.doc;
                })
                .ToList();

            return merged;
        }

        private List<string> GenerateQueries(string originalQuery)
        {
            var queries = new List<string> { originalQuery };

            for (int i = 1; i < _numQueries; i++)
            {
                queries.Add($"{originalQuery} variation {i}");
            }

            return queries;
        }
    }
}
