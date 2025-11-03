using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// BM25 (Best Matching 25) retrieval algorithm for sparse keyword-based search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class BM25Retriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly T _k1;
        private readonly T _b;
        private readonly Dictionary<string, Dictionary<string, T>> _termFrequencies;
        private readonly Dictionary<string, T> _documentLengths;
        private T _avgDocLength;

        public BM25Retriever(IDocumentStore<T> documentStore, int defaultTopK = 5, double k1 = 1.5, double b = 0.75) 
            : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));

            _documentStore = documentStore;
            _k1 = NumOps.FromDouble(k1);
            _b = NumOps.FromDouble(b);
            _termFrequencies = new Dictionary<string, Dictionary<string, T>>();
            _documentLengths = new Dictionary<string, T>();
            _avgDocLength = NumOps.Zero;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            var candidates = _documentStore.GetSimilar(
                new AiDotNet.LinearAlgebra.Vector<T>(new T[Math.Max(1, _documentStore.VectorDimension)]),
                _documentStore.DocumentCount
            );

            foreach (var doc in candidates)
            {
                if (!MatchesFilters(doc, metadataFilters))
                    continue;

                var score = NumOps.Zero;
                
                foreach (var term in queryTerms)
                {
                    var termScore = CalculateBM25Term(doc.Id, term);
                    score = NumOps.Add(score, termScore);
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

        private T CalculateBM25Term(string docId, string term)
        {
            if (!_termFrequencies.ContainsKey(docId) || !_termFrequencies[docId].ContainsKey(term))
                return NumOps.Zero;

            var tf = _termFrequencies[docId][term];
            var docLength = _documentLengths.ContainsKey(docId) ? _documentLengths[docId] : NumOps.Zero;

            var numerator = NumOps.Multiply(tf, NumOps.Add(_k1, NumOps.One));
            var denominator = NumOps.Add(tf, NumOps.Multiply(_k1, 
                NumOps.Subtract(NumOps.One, NumOps.Add(_b, NumOps.Multiply(_b, NumOps.Divide(docLength, _avgDocLength))))));

            return NumOps.Divide(numerator, denominator);
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
