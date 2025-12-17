using System;
using System.Collections.Generic;
using System.Linq;

using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression
{
    /// <summary>
    /// Document summarizer for creating concise summaries of retrieved content.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class DocumentSummarizer<T> : ContextCompressorBase<T>
    {
        private readonly int _maxSummaryLength;
        private readonly string _llmEndpoint;
        private readonly string _apiKey;

        /// <summary>
        /// Initializes a new instance of the <see cref="DocumentSummarizer{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="maxSummaryLength">The maximum length of the summary in characters.</param>
        /// <param name="llmEndpoint">The LLM API endpoint.</param>
        /// <param name="apiKey">The API key for the LLM service.</param>
        public DocumentSummarizer(
            INumericOperations<T> numericOperations,
            int maxSummaryLength = 500,
            string llmEndpoint = "",
            string apiKey = "")
        {
            if (numericOperations == null)
                throw new ArgumentNullException(nameof(numericOperations));

            _maxSummaryLength = maxSummaryLength > 0
                ? maxSummaryLength
                : throw new ArgumentOutOfRangeException(nameof(maxSummaryLength));
            _llmEndpoint = llmEndpoint;
            _apiKey = apiKey;
        }

        /// <summary>
        /// Compresses documents by summarizing their content with query-aware sentence selection.
        /// </summary>
        protected override List<Document<T>> CompressCore(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null)
        {
            var summarized = new List<Document<T>>();
            var queryTerms = Tokenize(query.ToLowerInvariant());

            foreach (var doc in documents)
            {
                var summary = SummarizeText(doc.Content, queryTerms);
                var summarizedDoc = new Document<T>(doc.Id, summary)
                {
                    Metadata = doc.Metadata != null ? new Dictionary<string, object>(doc.Metadata) : new Dictionary<string, object>(),
                    RelevanceScore = doc.RelevanceScore,
                    HasRelevanceScore = doc.HasRelevanceScore
                };
                summarized.Add(summarizedDoc);
            }

            return summarized;
        }

        /// <summary>
        /// Summarizes a list of documents.
        /// </summary>
        /// <param name="documents">The documents to summarize.</param>
        /// <returns>A list of summarized documents.</returns>
        public List<Document<T>> Summarize(List<Document<T>> documents)
        {
            if (documents == null) throw new ArgumentNullException(nameof(documents));

            var summarized = new List<Document<T>>();

            foreach (var doc in documents)
            {
                var summary = SummarizeText(doc.Content);
                var summarizedDoc = new Document<T>(doc.Id, summary)
                {
                    Metadata = doc.Metadata != null ? new Dictionary<string, object>(doc.Metadata) : new Dictionary<string, object>(),
                    RelevanceScore = doc.RelevanceScore,
                    HasRelevanceScore = doc.HasRelevanceScore
                };
                summarized.Add(summarizedDoc);
            }

            return summarized;
        }

        /// <summary>
        /// Summarizes text to a maximum length with query-aware sentence selection.
        /// </summary>
        /// <param name="text">The text to summarize.</param>
        /// <param name="queryTerms">Optional query terms to prioritize relevant content.</param>
        /// <returns>The summarized text.</returns>
        public string SummarizeText(string text, List<string>? queryTerms = null)
        {
            if (string.IsNullOrEmpty(text)) return text;

            if (text.Length <= _maxSummaryLength)
            {
                return text;
            }

            var sentences = SplitIntoSentences(text);
            var importantSentences = ExtractImportantSentences(sentences, queryTerms);

            const string ellipsis = "...";
            int ellipsisLength = ellipsis.Length;

            var summary = new System.Text.StringBuilder();
            foreach (var sentence in importantSentences)
            {
                if (summary.Length + sentence.Length > _maxSummaryLength)
                {
                    // If we haven't added anything yet and the first sentence is too long,
                    // truncate it to fit within maxSummaryLength (accounting for ellipsis)
                    if (summary.Length == 0 && sentence.Length > _maxSummaryLength)
                    {
                        // For very small limits, just truncate without ellipsis
                        if (_maxSummaryLength <= ellipsisLength)
                        {
                            return sentence.Substring(0, _maxSummaryLength);
                        }
                        return sentence.Substring(0, _maxSummaryLength - ellipsisLength).Trim() + ellipsis;
                    }
                    break;
                }
                summary.Append(sentence).Append(" ");
            }

            var result = summary.ToString().Trim();
            // If result is empty (all sentences too long), return truncated first sentence
            if (string.IsNullOrEmpty(result) && importantSentences.Any())
            {
                var firstSentence = importantSentences.First();
                // For very small limits, just truncate without ellipsis
                if (_maxSummaryLength <= ellipsisLength)
                {
                    return firstSentence.Substring(0, Math.Min(firstSentence.Length, _maxSummaryLength));
                }
                int truncateLength = Math.Min(firstSentence.Length, _maxSummaryLength - ellipsisLength);
                return firstSentence.Substring(0, truncateLength).Trim() + ellipsis;
            }
            return result;
        }

        private List<string> ExtractImportantSentences(List<string> sentences, List<string>? queryTerms = null)
        {
            var scored = new List<(string sentence, double score)>();

            foreach (var sentence in sentences)
            {
                var importance = ComputeImportance(sentence, sentences, queryTerms);
                scored.Add((sentence, importance));
            }

            return scored
                .OrderByDescending(x => x.score)
                .Select(x => x.sentence)
                .ToList();
        }

        private double ComputeImportance(string sentence, List<string> allSentences, List<string>? queryTerms = null)
        {
            var tokens = Tokenize(sentence);
            var uniqueTokens = tokens.Distinct().Count();
            var length = sentence.Length;

            var positionScore = allSentences.IndexOf(sentence) == 0 ? 1.5 : 1.0;

            var importance = (uniqueTokens * 0.5) + (Math.Min(length, 200) / 200.0 * 0.5);
            importance *= positionScore;

            // Boost sentences containing query terms
            if (queryTerms != null && queryTerms.Count > 0)
            {
                var sentenceLower = sentence.ToLowerInvariant();
                var matchCount = queryTerms.Count(qt => sentenceLower.Contains(qt));
                if (matchCount > 0)
                {
                    importance *= (1.0 + (matchCount * 0.5)); // Boost by 50% per matching query term
                }
            }

            return importance;
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
