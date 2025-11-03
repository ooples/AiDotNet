using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression
{
    /// <summary>
    /// Document summarizer for creating concise summaries of retrieved content.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class DocumentSummarizer<T>
    {
        private readonly INumericOperations<T> _numOps;
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
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _maxSummaryLength = maxSummaryLength > 0
                ? maxSummaryLength
                : throw new ArgumentOutOfRangeException(nameof(maxSummaryLength));
            _llmEndpoint = llmEndpoint;
            _apiKey = apiKey;
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
                var summarizedDoc = new Document<T>
                {
                    Id = doc.Id,
                    Content = summary,
                    Metadata = doc.Metadata,
                    RelevanceScore = doc.RelevanceScore
                };
                summarized.Add(summarizedDoc);
            }

            return summarized;
        }

        /// <summary>
        /// Summarizes text to a maximum length.
        /// </summary>
        /// <param name="text">The text to summarize.</param>
        /// <returns>The summarized text.</returns>
        public string SummarizeText(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            if (text.Length <= _maxSummaryLength)
            {
                return text;
            }

            var sentences = SplitIntoSentences(text);
            var importantSentences = ExtractImportantSentences(sentences);

            var summary = string.Empty;
            foreach (var sentence in importantSentences)
            {
                if (summary.Length + sentence.Length > _maxSummaryLength)
                {
                    break;
                }
                summary += sentence + " ";
            }

            return summary.Trim();
        }

        private List<string> ExtractImportantSentences(List<string> sentences)
        {
            var scored = new List<(string sentence, double score)>();

            foreach (var sentence in sentences)
            {
                var importance = ComputeImportance(sentence, sentences);
                scored.Add((sentence, importance));
            }

            return scored
                .OrderByDescending(x => x.score)
                .Select(x => x.sentence)
                .ToList();
        }

        private double ComputeImportance(string sentence, List<string> allSentences)
        {
            var tokens = Tokenize(sentence);
            var uniqueTokens = tokens.Distinct().Count();
            var length = sentence.Length;

            var positionScore = allSentences.IndexOf(sentence) == 0 ? 1.5 : 1.0;

            var importance = (uniqueTokens * 0.5) + (Math.Min(length, 200) / 200.0 * 0.5);
            importance *= positionScore;

            return importance;
        }

        private List<string> SplitIntoSentences(string text)
        {
            var sentences = new List<string>();
            var sentenceEndings = new[] { ". ", "! ", "? ", ".\n", "!\n", "?\n" };
            var currentSentence = string.Empty;

            for (int i = 0; i < text.Length; i++)
            {
                currentSentence += text[i];

                foreach (var ending in sentenceEndings)
                {
                    if (currentSentence.EndsWith(ending))
                    {
                        sentences.Add(currentSentence.Trim());
                        currentSentence = string.Empty;
                        break;
                    }
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
