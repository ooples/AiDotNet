using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class TFIDFRetriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly Dictionary<string, Dictionary<string, T>> _tfidf;
        private readonly Dictionary<string, T> _idf;

        public TFIDFRetriever(IDocumentStore<T> documentStore, int defaultTopK = 5) : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));
                
            _documentStore = documentStore;
            _tfidf = new Dictionary<string, Dictionary<string, T>>();
            _idf = new Dictionary<string, T>();
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            var candidates = _documentStore.GetSimilar(
                new AiDotNet.LinearAlgebra.Vector<T>(new T[_documentStore.VectorDimension]), 
                _documentStore.DocumentCount
            );

            foreach (var doc in candidates)
            {
                if (!MatchesFilters(doc, metadataFilters))
                    continue;

                var score = NumOps.Zero;

                if (_tfidf.ContainsKey(doc.Id))
                {
                    var docTfidf = _tfidf[doc.Id];
                    foreach (var term in queryTerms)
                    {
                        if (docTfidf.ContainsKey(term))
                        {
                            score = NumOps.Add(score, docTfidf[term]);
                        }
                    }
                }

                scores[doc.Id] = score;
            }

            var results = scores
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .Select(kv =>
                {
                    var doc = candidates.FirstOrDefault(d => d.Id == kv.Key);
                    if (doc != null)
                    {
                        doc.RelevanceScore = kv.Value;
                        doc.HasRelevanceScore = true;
                    }
                    return doc;
                })
                .Where(d => d != null)
                .ToList();

            return results;
        }

        private List<string> Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }

        private bool MatchesFilters(Document<T> document, Dictionary<string, object> filters)
        {
            if (filters.Count == 0)
                return true;

            foreach (var filter in filters)
            {
                if (!document.Metadata.TryGetValue(filter.Key, out var value))
                    return false;

                if (!filter.Value.Equals(value))
                    return false;
            }

            return true;
        }
    }
}
