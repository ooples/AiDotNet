using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression
{
    /// <summary>
    /// LLM-based context compression to reduce token usage while preserving key information.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LLMContextCompressor<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly string _llmEndpoint;
        private readonly string _apiKey;
        private readonly double _compressionRatio;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMContextCompressor{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="compressionRatio">The target compression ratio (0.0 to 1.0).</param>
        /// <param name="llmEndpoint">The LLM API endpoint.</param>
        /// <param name="apiKey">The API key for the LLM service.</param>
        public LLMContextCompressor(
            INumericOperations<T> numericOperations,
            double compressionRatio = 0.5,
            string llmEndpoint = "",
            string apiKey = "")
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _compressionRatio = compressionRatio >= 0 && compressionRatio <= 1
                ? compressionRatio
                : throw new ArgumentOutOfRangeException(nameof(compressionRatio), "Compression ratio must be between 0 and 1");
            _llmEndpoint = llmEndpoint;
            _apiKey = apiKey;
        }

        /// <summary>
        /// Compresses a list of documents while preserving relevance to the query.
        /// </summary>
        /// <param name="query">The query context.</param>
        /// <param name="documents">The documents to compress.</param>
        /// <returns>A list of compressed documents.</returns>
        public List<Document<T>> Compress(string query, List<Document<T>> documents)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (documents == null) throw new ArgumentNullException(nameof(documents));

            var compressed = new List<Document<T>>();

            foreach (var doc in documents)
            {
                var compressedContent = CompressText(query, doc.Content);
                var compressedDoc = new Document<T>
                {
                    Id = doc.Id,
                    Content = compressedContent,
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
            var sentences = new List<string>();
            var sentenceEndings = new[] { ". ", "! ", "? ", ".\n", "!\n", "?\n" };
            var currentSentence = string.Empty;

            for (int i = 0; i < text.Length; i++)
            {
                currentSentence += text[i];

                var matchedEnding = sentenceEndings.FirstOrDefault(ending => currentSentence.EndsWith(ending));
                if (matchedEnding != null)
                {
                    sentences.Add(currentSentence.Trim());
                    currentSentence = string.Empty;
                }
            }

            if (!string.IsNullOrWhiteSpace(currentSentence))
            {
                sentences.Add(currentSentence.Trim());
            }

            return sentences;
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
