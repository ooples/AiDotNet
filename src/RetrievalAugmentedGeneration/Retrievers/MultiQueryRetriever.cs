
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

        /// <inheritdoc />
        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            // CombSUM over the generated queries: a document's score is the sum of its scores for every query
            // that retrieved it (the latest returned instance of the document is kept).
            var position = new Dictionary<string, int>();
            var documents = new List<Document<T>>();
            var hits = new List<int>();
            var hitScores = new List<T>();

            foreach (var q in GenerateQueries(query))
            {
                foreach (var doc in _baseRetriever.Retrieve(q, topK, metadataFilters).Where(d => d.HasRelevanceScore))
                {
                    if (position.TryGetValue(doc.Id, out var index))
                    {
                        documents[index] = doc;
                    }
                    else
                    {
                        index = documents.Count;
                        position[doc.Id] = index;
                        documents.Add(doc);
                    }

                    hits.Add(index);
                    hitScores.Add(doc.RelevanceScore);
                }
            }

            if (hits.Count == 0)
                return new List<Document<T>>();

            // Segmented sum per document: linear in the hits, no [hits x documents] membership matrix.
            var totals = Engine.ScatterAdd(
                new Tensor<T>(hitScores.ToArray(), [hitScores.Count]), new Tensor<int>(hits.ToArray(), [hits.Count]), 0, documents.Count).ToArray();

            return documents
                .Select((doc, i) => (doc, score: totals[i]))
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    x.doc.HasRelevanceScore = true;
                    return x.doc;
                })
                .ToList();
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
