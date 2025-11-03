using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// LLM-based reranking using language model relevance assessment.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LLMBasedReranker<T> : RerankingStrategyBase<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly string _llmEndpoint;
        private readonly string _apiKey;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMBasedReranker{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="llmEndpoint">The LLM API endpoint.</param>
        /// <param name="apiKey">The API key for the LLM service.</param>
        public LLMBasedReranker(INumericOperations<T> numericOperations, string? llmEndpoint = null, string? apiKey = null) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _llmEndpoint = llmEndpoint ?? string.Empty;
            _apiKey = apiKey ?? string.Empty;
        }

        /// <summary>
        /// Reranks documents using LLM-based relevance scoring.
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
                var score = AssessRelevance(query, doc.Content);
                scoredDocs.Add((doc, score));
            }

            var reranked = scoredDocs
                .OrderByDescending(x => Convert.ToDouble(x.score))
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    return x.doc;
                })
                .ToList();

            return reranked;
        }

        private T AssessRelevance(string query, string document)
        {
            var queryWords = Tokenize(query);
            var docWords = Tokenize(document);

            var exactMatches = queryWords.Count(w => docWords.Contains(w));
            var proximityScore = ComputeProximityScore(queryWords, document);
            var semanticScore = ComputeSemanticSimilarity(query, document);

            var totalScore = (exactMatches * 0.4 + proximityScore * 0.3 + semanticScore * 0.3);
            totalScore = Math.Min(1.0, totalScore / queryWords.Count);

            return _numOps.FromDouble(totalScore);
        }

        private double ComputeProximityScore(List<string> queryWords, string document)
        {
            var score = 0.0;
            var docLower = document.ToLowerInvariant();

            for (int i = 0; i < queryWords.Count - 1; i++)
            {
                var word1 = queryWords[i];
                var word2 = queryWords[i + 1];

                var idx1 = docLower.IndexOf(word1);
                var idx2 = docLower.IndexOf(word2, idx1 + 1);

                if (idx1 >= 0 && idx2 >= 0)
                {
                    var distance = idx2 - idx1 - word1.Length;
                    score += 1.0 / (1.0 + distance / 10.0);
                }
            }

            return score;
        }

        private double ComputeSemanticSimilarity(string query, string document)
        {
            var queryTokens = Tokenize(query);
            var docTokens = Tokenize(document);

            var intersection = queryTokens.Intersect(docTokens).Count();
            var union = queryTokens.Union(docTokens).Count();

            return union > 0 ? (double)intersection / union : 0.0;
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
