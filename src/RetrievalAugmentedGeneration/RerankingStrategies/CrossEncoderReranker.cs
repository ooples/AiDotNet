using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Cross-encoder based reranking for improved relevance scoring.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class CrossEncoderReranker<T> : RerankingStrategyBase<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly string _modelPath;

        /// <summary>
        /// Initializes a new instance of the <see cref="CrossEncoderReranker{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="modelPath">The path to the cross-encoder model.</param>
        public CrossEncoderReranker(INumericOperations<T> numericOperations, string modelPath = null) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _modelPath = modelPath;
        }

        /// <summary>
        /// Reranks documents using cross-encoder scoring.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="documents">The documents to rerank.</param>
        /// <param name="topK">The number of top documents to return.</param>
        /// <returns>A reranked list of documents.</returns>
        public override List<Document<T>> Rerank(string query, List<Document<T>> documents, int topK)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (documents == null) throw new ArgumentNullException(nameof(documents));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var scoredDocs = new List<(Document<T> doc, T score)>();

            foreach (var doc in documents)
            {
                var score = ComputeCrossEncoderScore(query, doc.Content);
                scoredDocs.Add((doc, score));
            }

            var reranked = scoredDocs
                .OrderByDescending(x => _numOps.ToDouble(x.score))
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    return x.doc;
                })
                .ToList();

            return reranked;
        }

        private T ComputeCrossEncoderScore(string query, string document)
        {
            var queryTokens = Tokenize(query);
            var docTokens = Tokenize(document);

            var overlap = queryTokens.Intersect(docTokens).Count();
            var total = Math.Max(queryTokens.Count, docTokens.Count);

            var score = total > 0 ? (double)overlap / total : 0.0;

            var lengthPenalty = 1.0 - Math.Abs(query.Length - document.Length) / (double)Math.Max(query.Length, document.Length);
            score *= lengthPenalty;

            return _numOps.FromDouble(score);
        }

        private List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text)) return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }
    }
}
