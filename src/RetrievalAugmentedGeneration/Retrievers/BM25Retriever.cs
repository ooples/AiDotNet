
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

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

        public BM25Retriever(IDocumentStore<T> documentStore, int defaultTopK = 5, double k1 = 1.5, double b = 0.75)
            : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));

            _documentStore = documentStore;
            _k1 = NumOps.FromDouble(k1);
            _b = NumOps.FromDouble(b);
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            var candidates = _documentStore.GetAll().ToList();
            var corpusStats = BuildCorpusStatistics(candidates);

            // Build document lookup dictionary for O(1) access
            var documentLookup = candidates.ToDictionary(d => d.Id);

            foreach (var doc in candidates.Where(d => MatchesFilters(d, metadataFilters)))
            {
                var score = NumOps.Zero;

                foreach (var term in queryTerms)
                {
                    var termScore = CalculateBM25Term(doc.Id, term, corpusStats);
                    score = NumOps.Add(score, termScore);
                }

                scores[doc.Id] = score;
            }

            var results = scores
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .Select(kv =>
                {
                    if (documentLookup.TryGetValue(kv.Key, out var doc))
                    {
                        doc.RelevanceScore = kv.Value;
                        doc.HasRelevanceScore = true;
                        return doc;
                    }
                    return null;
                })
                .Where(d => d != null)
                .Cast<Document<T>>();

            return results;
        }

        private T CalculateBM25Term(string docId, string term, CorpusStatistics corpusStats)
        {
            if (!corpusStats.TermFrequencies.TryGetValue(docId, out var termFreqs) || !termFreqs.TryGetValue(term, out var tf))
                return NumOps.Zero;

            var docLength = corpusStats.DocumentLengths.TryGetValue(docId, out var length) ? length : NumOps.Zero;

            var idf = CalculateIDF(term, corpusStats);

            var numerator = NumOps.Multiply(tf, NumOps.Add(_k1, NumOps.One));
            var lengthNorm = NumOps.Add(
                NumOps.Subtract(NumOps.One, _b),
                NumOps.Multiply(_b, NumOps.Divide(docLength, corpusStats.AvgDocLength)));
            var denominator = NumOps.Add(tf, NumOps.Multiply(_k1, lengthNorm));

            return NumOps.Multiply(idf, NumOps.Divide(numerator, denominator));
        }

        private T CalculateIDF(string term, CorpusStatistics corpusStats)
        {
            if (!corpusStats.DocumentFrequencies.ContainsKey(term) || corpusStats.TotalDocuments == 0)
                return NumOps.Zero;

            var df = corpusStats.DocumentFrequencies[term];
            if (df == 0)
                return NumOps.Zero;

            var numerator = NumOps.FromDouble(corpusStats.TotalDocuments - df + 0.5);
            var denominator = NumOps.FromDouble(df + 0.5);
            var ratio = NumOps.Divide(numerator, denominator);

            return NumOps.FromDouble(Math.Log(Convert.ToDouble(ratio)));
        }

        private class CorpusStatistics
        {
            public Dictionary<string, Dictionary<string, T>> TermFrequencies { get; set; } = new();
            public Dictionary<string, T> DocumentLengths { get; set; } = new();
            public Dictionary<string, int> DocumentFrequencies { get; set; } = new();
            public int TotalDocuments { get; set; }
            public T AvgDocLength { get; set; }

            public CorpusStatistics()
            {
                AvgDocLength = NumOps.One;
            }
        }

        private CorpusStatistics BuildCorpusStatistics(List<Document<T>> documents)
        {
            var stats = new CorpusStatistics();

            if (documents == null || documents.Count == 0)
            {
                return stats;
            }

            stats.TotalDocuments = documents.Count;

            var totalLength = NumOps.Zero;

            foreach (var doc in documents)
            {
                var terms = Tokenize(doc.Content);
                var termCounts = new Dictionary<string, T>();

                foreach (var term in terms)
                {
                    if (termCounts.TryGetValue(term, out var count))
                        termCounts[term] = NumOps.Add(count, NumOps.One);
                    else
                        termCounts[term] = NumOps.One;

                    if (!stats.DocumentFrequencies.ContainsKey(term))
                        stats.DocumentFrequencies[term] = 0;
                }

                foreach (var term in termCounts.Keys.Distinct())
                {
                    stats.DocumentFrequencies[term]++;
                }

                stats.TermFrequencies[doc.Id] = termCounts;
                var docLength = NumOps.FromDouble(terms.Count);
                stats.DocumentLengths[doc.Id] = docLength;
                totalLength = NumOps.Add(totalLength, docLength);
            }

            stats.AvgDocLength = NumOps.Divide(totalLength, NumOps.FromDouble(stats.TotalDocuments));

            if (!NumOps.GreaterThan(stats.AvgDocLength, NumOps.Zero))
                stats.AvgDocLength = NumOps.One;

            return stats;
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
