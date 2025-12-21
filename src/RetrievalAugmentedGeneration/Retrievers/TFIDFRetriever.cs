
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy with cached statistics.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// Implements production-ready TF-IDF retrieval with intelligent caching to avoid recomputing
    /// statistics on every query. The cache is automatically invalidated when the document count changes,
    /// ensuring accuracy while maximizing performance.
    /// </para>
    /// <para><b>For Beginners:</b> TF-IDF ranks documents by how unique and frequent terms are.
    /// 
    /// This implementation uses a cache to avoid recalculating term statistics on every search,
    /// dramatically improving performance for repeated queries. The cache is automatically refreshed
    /// when documents are added or removed.
    /// </para>
    /// </remarks>
    public class TFIDFRetriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly object _cacheLock = new object();

        private Dictionary<string, Dictionary<string, T>>? _cachedTfidf;
        private Dictionary<string, T>? _cachedIdf;
        private int _cachedDocumentCount;

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

            var candidates = _documentStore.GetAll();
            var candidatesList = candidates.ToList();
            var candidatesById = candidatesList.ToDictionary(d => d.Id);

            // Use cached TF-IDF or build if cache is stale
            var tfidf = GetOrBuildTFIDF(candidatesList);

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
                .Select(kv => new { DocId = kv.Key, Score = Convert.ToDouble(kv.Value) })
                .OrderByDescending(x => x.Score)
                .Take(topK)
                .Select(x =>
                {
                    if (candidatesById.TryGetValue(x.DocId, out var doc))
                    {
                        doc.RelevanceScore = NumOps.FromDouble(x.Score);
                        doc.HasRelevanceScore = true;
                        return doc;
                    }
                    return null;
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

        private Dictionary<string, Dictionary<string, T>> GetOrBuildTFIDF(List<Document<T>> documents)
        {
            var currentDocCount = documents.Count;

            lock (_cacheLock)
            {
                // Return cached statistics if document count hasn't changed
                if (_cachedTfidf != null && _cachedDocumentCount == currentDocCount)
                {
                    return _cachedTfidf;
                }

                // Build new statistics and cache them
                var (tfidf, idf) = BuildTFIDFStatistics(documents);
                _cachedTfidf = tfidf;
                _cachedIdf = idf;
                _cachedDocumentCount = currentDocCount;

                return tfidf;
            }
        }

        private (Dictionary<string, Dictionary<string, T>> tfidf, Dictionary<string, T> idf) BuildTFIDFStatistics(List<Document<T>> documents)
        {
            if (documents == null || documents.Count == 0)
                return (new Dictionary<string, Dictionary<string, T>>(), new Dictionary<string, T>());

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

            return (tfidf, idf);
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
