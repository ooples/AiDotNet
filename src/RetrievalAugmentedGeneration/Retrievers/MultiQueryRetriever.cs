using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Multi-query retriever that generates multiple queries and merges results.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class MultiQueryRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly INumericOperations<T> _numOps;
        private readonly int _numQueries;

        /// <summary>
        /// Initializes a new instance of the <see cref="MultiQueryRetriever{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="baseRetriever">The base retriever to use for each query.</param>
        /// <param name="numQueries">The number of queries to generate.</param>
        public MultiQueryRetriever(
            INumericOperations<T> numericOperations,
            IRetriever<T> baseRetriever,
            int numQueries = 3) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));
            _numQueries = numQueries > 0 ? numQueries : throw new ArgumentOutOfRangeException(nameof(numQueries));
        }

        /// <summary>
        /// Retrieves documents using multiple query variations.
        /// </summary>
        /// <param name="query">The original query string.</param>
        /// <param name="topK">The number of documents to retrieve.</param>
        /// <returns>A merged list of the most relevant documents.</returns>
        public override List<Document<T>> Retrieve(string query, int topK)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var queries = GenerateQueries(query);
            var allResults = new Dictionary<string, (Document<T> doc, T score)>();

            foreach (var q in queries)
            {
                var results = _baseRetriever.Retrieve(q, topK);

                foreach (var doc in results)
                {
                    if (allResults.ContainsKey(doc.Id))
                    {
                        var currentScore = allResults[doc.Id].score;
                        allResults[doc.Id] = (doc, _numOps.Add(currentScore, doc.RelevanceScore));
                    }
                    else
                    {
                        allResults[doc.Id] = (doc, doc.RelevanceScore);
                    }
                }
            }

            var finalResults = allResults.Values
                .OrderByDescending(x => _numOps.ToDouble(x.score))
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    return x.doc;
                })
                .ToList();

            return finalResults;
        }

        private List<string> GenerateQueries(string query)
        {
            var queries = new List<string> { query };

            for (int i = 1; i < _numQueries; i++)
            {
                queries.Add(RephaseQuery(query, i));
            }

            return queries;
        }

        private string RephaseQuery(string query, int variant)
        {
            var words = query.Split(' ');

            switch (variant % 3)
            {
                case 0:
                    return string.Join(" ", words.Reverse());
                case 1:
                    return $"What is {query}?";
                case 2:
                    return $"Explain {query}";
                default:
                    return query;
            }
        }
    }
}
