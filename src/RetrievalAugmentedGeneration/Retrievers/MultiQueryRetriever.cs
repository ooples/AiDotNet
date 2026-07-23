
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Multi-query retriever that generates multiple query variations and merges results.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class MultiQueryRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly int _numQueries;
        private readonly ITextGenerator? _generator;

        /// <param name="baseRetriever">The underlying retriever run once per query variation.</param>
        /// <param name="numQueries">Total number of queries (original + variations) to retrieve with.</param>
        /// <param name="defaultTopK">Default number of documents to return.</param>
        /// <param name="generator">
        /// Optional real text generator. When provided, the query variations are LLM-generated; otherwise a
        /// deterministic template fallback is used (never the old "{query} variation N" placeholder).
        /// </param>
        public MultiQueryRetriever(IRetriever<T> baseRetriever, int numQueries = 3, int defaultTopK = 5, ITextGenerator? generator = null) : base(defaultTopK)
        {
            if (baseRetriever == null)
                throw new ArgumentNullException(nameof(baseRetriever));
            if (numQueries <= 0)
                throw new ArgumentException("Number of queries must be positive", nameof(numQueries));

            _baseRetriever = baseRetriever;
            _numQueries = numQueries;
            _generator = generator;
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
            // Reuse the (de-stubbed) LLM query expander: real LLM variations when a generator is supplied,
            // deterministic template variations otherwise. ExpandQuery prepends the original, so request
            // (_numQueries - 1) expansions and cap the result at _numQueries.
            var expander = new QueryExpansion.LLMQueryExpansion(
                numExpansions: Math.Max(1, _numQueries - 1), generator: _generator);
            return expander.ExpandQuery(originalQuery).Take(_numQueries).ToList();
        }
    }
}
