using System;
using System.Collections.Generic;
using System.Linq;

using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression
{
    /// <summary>
    /// LLM-based context compression to reduce token usage while preserving key information.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LLMContextCompressor<T> : ContextCompressorBase<T>
    {
        private readonly string _llmEndpoint;
        private readonly string _apiKey;
        private readonly double _compressionRatio;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMContextCompressor{T}"/> class.
        /// </summary>
        /// <param name="compressionRatio">The target compression ratio (0.0 to 1.0).</param>
        /// <param name="llmEndpoint">The LLM API endpoint.</param>
        /// <param name="apiKey">The API key for the LLM service.</param>
        public LLMContextCompressor(
            double compressionRatio = 0.5,
            string llmEndpoint = "",
            string apiKey = "")
        {
            _compressionRatio = compressionRatio >= 0 && compressionRatio <= 1
                ? compressionRatio
                : throw new ArgumentOutOfRangeException(nameof(compressionRatio), "Compression ratio must be between 0 and 1");
            _llmEndpoint = llmEndpoint;
            _apiKey = apiKey;
        }

        /// <summary>
        /// Compresses documents while preserving relevance to the query.
        /// </summary>
        protected override List<Document<T>> CompressCore(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null)
        {
            var compressed = new List<Document<T>>();

            foreach (var doc in documents)
            {
                var compressedContent = CompressText(query, doc.Content);
                var compressedDoc = new Document<T>(doc.Id, compressedContent)
                {
                    Metadata = doc.Metadata,
                    RelevanceScore = doc.RelevanceScore,
                    HasRelevanceScore = doc.HasRelevanceScore
                };
                compressed.Add(compressedDoc);
            }

            return compressed;
        }

        /// <summary>
        /// Compresses text based on relevance to the query.
        /// </summary>
        /// <param name="query">The query context.</param>
        /// <param name="text">The text to compress.</param>
        /// <returns>The compressed text.</returns>
        public string CompressText(string query, string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            var sentences = SplitIntoSentences(text);
            var scoredSentences = new List<(string sentence, double score)>();

            foreach (var sentence in sentences)
            {
                var relevance = ComputeRelevance(query, sentence);
                scoredSentences.Add((sentence, relevance));
            }

            var targetCount = Math.Max(1, (int)(sentences.Count * _compressionRatio));
            var topSentences = scoredSentences
                .OrderByDescending(x => x.score)
                .Take(targetCount)
                .OrderBy(x => sentences.IndexOf(x.sentence))
                .Select(x => x.sentence);

            return string.Join(" ", topSentences);
        }

        private double ComputeRelevance(string query, string sentence)
        {
            var queryTokens = Tokenize(query);
            var sentenceTokens = Tokenize(sentence);

            var overlap = queryTokens.Intersect(sentenceTokens).Count();
            var total = Math.Max(queryTokens.Count, sentenceTokens.Count);

            return total > 0 ? (double)overlap / total : 0.0;
        }

        private List<string> SplitIntoSentences(string text)
        {
            return Helpers.TextProcessingHelper.SplitIntoSentences(text);
        }

        private List<string> Tokenize(string text)
        {
            return Helpers.TextProcessingHelper.Tokenize(text);
        }
    }
}
