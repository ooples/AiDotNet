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
        private readonly Dictionary<string, int> _documentFrequencies;
        private int _totalDocuments;
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
            _documentFrequencies = new Dictionary<string, int>();
            _avgDocLength = NumOps.Zero;
            _totalDocuments = 0;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            var candidates = _documentStore.GetSimilar(
                new AiDotNet.LinearAlgebra.Vector<T>(new T[Math.Max(1, _documentStore.VectorDimension)]),
                _documentStore.DocumentCount
            );

            var candidatesList = candidates.ToList();
            BuildCorpusStatistics(candidatesList);

            foreach (var doc in candidatesList)
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

            var idf = CalculateIDF(term);

            var numerator = NumOps.Multiply(tf, NumOps.Add(_k1, NumOps.One));
            var lengthNorm = NumOps.Add(
                NumOps.Subtract(NumOps.One, _b),
                NumOps.Multiply(_b, NumOps.Divide(docLength, _avgDocLength)));
            var denominator = NumOps.Add(tf, NumOps.Multiply(_k1, lengthNorm));

            return NumOps.Multiply(idf, NumOps.Divide(numerator, denominator));
        }

        private T CalculateIDF(string term)
        {
            if (!_documentFrequencies.ContainsKey(term) || _totalDocuments == 0)
                return NumOps.Zero;

            var df = _documentFrequencies[term];
            if (df == 0)
                return NumOps.Zero;

            var numerator = NumOps.FromDouble(_totalDocuments - df + 0.5);
            var denominator = NumOps.FromDouble(df + 0.5);
            var ratio = NumOps.Divide(numerator, denominator);
            
            return NumOps.FromDouble(Math.Log(Convert.ToDouble(ratio)));
        }

        private void BuildCorpusStatistics(List<Document<T>> documents)
        {
            if (documents == null || documents.Count == 0)
            {
                _avgDocLength = NumOps.One;
                _totalDocuments = 0;
                return;
            }

            _termFrequencies.Clear();
            _documentLengths.Clear();
            _documentFrequencies.Clear();
            _totalDocuments = documents.Count;

            var totalLength = NumOps.Zero;

            foreach (var doc in documents)
            {
                var terms = Tokenize(doc.Content);
                var termCounts = new Dictionary<string, T>();
                
                foreach (var term in terms)
                {
                    if (termCounts.ContainsKey(term))
                        termCounts[term] = NumOps.Add(termCounts[term], NumOps.One);
                    else
                        termCounts[term] = NumOps.One;

                    if (!_documentFrequencies.ContainsKey(term))
                        _documentFrequencies[term] = 0;
                }

                foreach (var term in termCounts.Keys.Distinct())
                {
                    _documentFrequencies[term]++;
                }

                _termFrequencies[doc.Id] = termCounts;
                var docLength = NumOps.FromDouble(terms.Count);
                _documentLengths[doc.Id] = docLength;
                totalLength = NumOps.Add(totalLength, docLength);
            }

            _avgDocLength = NumOps.Divide(totalLength, NumOps.FromDouble(_totalDocuments));
            
            if (!NumOps.GreaterThan(_avgDocLength, NumOps.Zero))
                _avgDocLength = NumOps.One;
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
