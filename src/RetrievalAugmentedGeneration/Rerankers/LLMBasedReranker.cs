using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// LLM-based reranking using language model relevance assessment.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Reranker)]
    [PipelineStage(PipelineStage.PostRetrieval)]
    public class LLMBasedReranker<T> : Rerankers.RerankerBase<T>
    {
        private readonly string _llmEndpoint;
        private readonly string _apiKey;
        private readonly ITextGenerator? _generator;

        /// <summary>
        /// Gets a value indicating whether this reranker modifies relevance scores.
        /// </summary>
        public override bool ModifiesScores => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMBasedReranker{T}"/> class.
        /// </summary>
        /// <param name="llmEndpoint">Optional LLM API endpoint (informational; the call goes through <paramref name="generator"/>).</param>
        /// <param name="apiKey">Optional API key (informational; the call goes through <paramref name="generator"/>).</param>
        /// <param name="generator">
        /// Optional real text generator. When provided, each document's relevance is scored by the LLM;
        /// when <c>null</c>, a lexical relevance heuristic is used as an offline fallback.
        /// </param>
        public LLMBasedReranker(string? llmEndpoint = null, string? apiKey = null, ITextGenerator? generator = null)
        {
            _llmEndpoint = llmEndpoint ?? string.Empty;
            _apiKey = apiKey ?? string.Empty;
            _generator = generator;
        }

        /// <summary>
        /// Reranks documents using LLM-based relevance scoring.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="documents">The documents to rerank.</param>
        /// <returns>A reranked list of documents.</returns>
        protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
        {
            var queryTokens = Tokenize(query);
            if (queryTokens.Count == 0)
            {
                return documents;
            }

            var scoredDocs = new List<(Document<T> doc, T score)>();

            foreach (var doc in documents)
            {
                var score = AssessRelevance(query, doc.Content);
                scoredDocs.Add((doc, score));
            }

            var reranked = scoredDocs
                .OrderByDescending(x => Convert.ToDouble(x.score))
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    x.doc.HasRelevanceScore = true;
                    return x.doc;
                })
                .ToList();

            return reranked;
        }

        private T AssessRelevance(string query, string document)
        {
            document ??= string.Empty;
            if (_generator != null)
            {
                var doc = document.Length > 2000 ? document.Substring(0, 2000) : document;
                var prompt =
                    $"On a scale of 0 to 10, how relevant is the document to the query? " +
                    $"Reply with only a single number.\n\nQuery: {query}\n\nDocument: {doc}\n\nRelevance (0-10):";
                var reply = _generator.Generate(prompt);
                var match = Regex.Match(reply ?? string.Empty, @"\d+(\.\d+)?");
                if (match.Success && double.TryParse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double raw))
                {
                    var normalized = Math.Min(1.0, Math.Max(0.0, raw / 10.0));
                    return NumOps.FromDouble(normalized);
                }
                // Unparseable reply: fall through to the lexical heuristic below.
            }

            var queryWords = Tokenize(query);
            var docWords = Tokenize(document);

            var exactMatches = queryWords.Count(w => docWords.Contains(w));
            var proximityScore = ComputeProximityScore(queryWords, document);
            var semanticScore = ComputeSemanticSimilarity(query, document);

            var totalScore = (exactMatches * 0.4 + proximityScore * 0.3 + semanticScore * 0.3);
            totalScore = Math.Min(1.0, totalScore / queryWords.Count);

            return NumOps.FromDouble(totalScore);
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
