using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RAG.ContextCompression
{
    public class DocumentSummarizer<T> : ContextCompressorBase<T>
        where T : struct, IComparable, IConvertible, IFormattable
    {
        private readonly int _maxSentences;
        private readonly bool _extractive;

        public DocumentSummarizer(int maxSentences = 5, bool extractive = true)
        {
            _maxSentences = maxSentences;
            _extractive = extractive;
        }

        protected override List<Document<T>> CompressCore(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null)
        {
            var summarized = new List<Document<T>>();
            
            foreach (var doc in documents)
            {
                var summary = _extractive 
                    ? ExtractiveSummarize(doc.Content, query)
                    : SimpleAbstractiveSummarize(doc.Content);
                
                summarized.Add(new Document<T>
                {
                    Id = doc.Id,
                    Content = summary,
                    Metadata = doc.Metadata,
                    Embedding = doc.Embedding
                });
            }
            
            return summarized;
        }

        private string ExtractiveSummarize(string content, string query)
        {
            var sentences = content.Split('.', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .ToList();
            
            if (sentences.Count <= _maxSentences)
            {
                return content;
            }

            var queryTerms = query.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var scoredSentences = sentences.Select(s => new
            {
                Sentence = s,
                Score = CalculateSentenceScore(s, queryTerms)
            })
            .OrderByDescending(x => x.Score)
            .Take(_maxSentences)
            .Select(x => x.Sentence)
            .ToList();
            
            return string.Join(". ", scoredSentences) + ".";
        }

        private string SimpleAbstractiveSummarize(string content)
        {
            var sentences = content.Split('.', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Take(_maxSentences)
                .ToList();
            
            return string.Join(". ", sentences) + ".";
        }

        private double CalculateSentenceScore(string sentence, string[] queryTerms)
        {
            var sentenceLower = sentence.ToLower();
            var matchCount = queryTerms.Count(term => sentenceLower.Contains(term));
            var lengthScore = 1.0 / (1.0 + Math.Abs(sentence.Length - 100) / 100.0);
            
            return matchCount * 2.0 + lengthScore;
        }
    }
}

