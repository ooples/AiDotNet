using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RAG.ContextCompression
{
    public class LLMContextCompressor<T> : ContextCompressorBase<T>
    {
        private readonly string _model;
        private readonly int _maxTokens;

        public LLMContextCompressor(string model = "gpt-3.5-turbo", int maxTokens = 500)
        {
            _model = model;
            _maxTokens = maxTokens;
        }

        protected override List<Document<T>> CompressCore(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null)
        {
            var compressed = new List<Document<T>>();
            
            foreach (var doc in documents)
            {
                var tokens = EstimateTokens(doc.Content);
                
                if (tokens <= _maxTokens)
                {
                    compressed.Add(doc);
                }
                else
                {
                    var summarized = SummarizeDocument(doc.Content, query);
                    compressed.Add(new Document<T>
                    {
                        Id = doc.Id,
                        Content = summarized,
                        Metadata = doc.Metadata,
                        Embedding = doc.Embedding
                    });
                }
            }
            
            return compressed;
        }

        private int EstimateTokens(string text)
        {
            return (int)Math.Ceiling(text.Length / 4.0);
        }

        private string SummarizeDocument(string content, string query)
        {
            var sentences = content.Split('.', StringSplitOptions.RemoveEmptyEntries);
            var relevantSentences = sentences
                .Where(s => ContainsQueryTerms(s, query))
                .Take(_maxTokens / 20)
                .ToList();
            
            return relevantSentences.Any() 
                ? string.Join(". ", relevantSentences) + "."
                : string.Join(". ", sentences.Take(3)) + ".";
        }

        private bool ContainsQueryTerms(string sentence, string query)
        {
            var queryTerms = query.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var sentenceLower = sentence.ToLower();
            
            return queryTerms.Any(term => sentenceLower.Contains(term));
        }
    }
}
