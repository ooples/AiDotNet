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

        public TFIDFRetriever(IDocumentStore<T> documentStore, int defaultTopK = 5) : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));
                
            _documentStore = documentStore;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            var candidates = _documentStore.GetSimilar(
                new AiDotNet.LinearAlgebra.Vector<T>(new T[_documentStore.VectorDimension]), 
                _documentStore.DocumentCount
            );

            var candidatesList = candidates.ToList();
            var tfidf = BuildTFIDFStatistics(candidatesList);

            foreach (var doc in candidatesList.Where(d => MatchesFilters(d, metadataFilters)))
            {
                var score = NumOps.Zero;

                if (tfidf.TryGetValue(doc.Id, out var docTfidf))
                {
                    foreach (var term in queryTerms.Where(t => docTfidf.ContainsKey(t)))
                    {
                        score = NumOps.Add(score, docTfidf[term]);
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
                .Cast<Document<T>>();

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

        private Dictionary<string, Dictionary<string, T>> BuildTFIDFStatistics(List<Document<T>> documents)
        {
            if (documents == null || documents.Count == 0)
                return new Dictionary<string, Dictionary<string, T>>();

            var termDocFreq = new Dictionary<string, int>();
            var docTermFreq = new Dictionary<string, Dictionary<string, int>>();

            foreach (var doc in documents)
            {
                var terms = Tokenize(doc.Content);
                var termCounts = new Dictionary<string, int>();
                
                foreach (var term in terms)
                {
                    if (termCounts.TryGetValue(term, out var count))
                        termCounts[term] = count + 1;
                    else
                        termCounts[term] = 1;
                }

                docTermFreq[doc.Id] = termCounts;

                foreach (var term in termCounts.Keys)
                {
                    if (termDocFreq.TryGetValue(term, out var docCount))
                        termDocFreq[term] = docCount + 1;
                    else
                        termDocFreq[term] = 1;
                }
            }

            var idf = new Dictionary<string, T>();
            foreach (var term in termDocFreq.Keys)
            {
                var df = termDocFreq[term];
                idf[term] = NumOps.FromDouble(Math.Log((double)documents.Count / (double)df));
            }

            var tfidf = new Dictionary<string, Dictionary<string, T>>();
            foreach (var doc in documents)
            {
                var termTfidf = new Dictionary<string, T>();
                var termCounts = docTermFreq[doc.Id];
                
                if (termCounts.Count == 0)
                {
                    tfidf[doc.Id] = termTfidf;
                    continue;
                }
                
                var maxFreq = termCounts.Values.Max();

                foreach (var termCount in termCounts)
                {
                    var tf = NumOps.FromDouble((double)termCount.Value / (double)maxFreq);
                    termTfidf[termCount.Key] = NumOps.Multiply(tf, idf[termCount.Key]);
                }

                tfidf[doc.Id] = termTfidf;
            }

            return tfidf;
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
