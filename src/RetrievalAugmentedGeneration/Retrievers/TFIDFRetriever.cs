
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
    /// statistics on every query. The cache is automatically invalidated when the corpus changes (a document
    /// added, removed or replaced, even when the document count stays the same), ensuring accuracy while
    /// maximizing performance.
    /// </para>
    /// <para><b>For Beginners:</b> TF-IDF ranks documents by how unique and frequent terms are.
    ///
    /// This implementation uses a cache to avoid recalculating term statistics on every search,
    /// dramatically improving performance for repeated queries. The cache is automatically refreshed
    /// when documents are added, removed or changed.
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class TFIDFRetriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly object _cacheLock = new object();

        private TermIndex? _cachedIndex;

        public TFIDFRetriever(IDocumentStore<T> documentStore, int defaultTopK = 5) : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));

            _documentStore = documentStore;
        }

        /// <inheritdoc />
        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var candidatesList = _documentStore.GetAll().ToList();

            // Use cached term statistics or build them if the cache is stale
            var index = GetOrBuildIndex(candidatesList);
            var indexScores = ScoreDocuments(Tokenize(query), index);
            var scoreById = new Dictionary<string, T>();
            for (int i = 0; i < index.DocumentIds.Count; i++)
                scoreById[index.DocumentIds[i]] = indexScores[i];

            return candidatesList
                .Where(d => MatchesFilters(d, metadataFilters))
                .Select(doc => (doc, score: scoreById.TryGetValue(doc.Id, out var score) ? score : NumOps.Zero))
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    x.doc.HasRelevanceScore = true;
                    return x.doc;
                })
                .ToList();
        }

        /// <summary>
        /// Σ over query terms (per occurrence) of tf · idf, with tf = count / max count in the document and
        /// idf = ln(N / df) (Salton and Buckley 1988), for every indexed document.
        /// </summary>
        internal static T[] ScoreDocuments(List<string> queryTerms, TermIndex index)
        {
            int documents = index.TermCounts.Count;
            var vocabulary = queryTerms.Distinct().Where(index.DocumentFrequency.ContainsKey).ToList();
            int terms = vocabulary.Count;
            if (documents == 0)
                return Array.Empty<T>();
            if (terms == 0)
                return new T[documents];

            var counts = new int[documents * terms];
            for (int d = 0; d < documents; d++)
            {
                for (int t = 0; t < terms; t++)
                {
                    counts[d * terms + t] = index.TermCounts[d].TryGetValue(vocabulary[t], out var count) ? count : 0;
                }
            }

            var queryCounts = vocabulary.Select(term => queryTerms.Count(q => q == term)).ToArray();
            var df = new Tensor<int>(vocabulary.Select(term => index.DocumentFrequency[term]).ToArray(), [terms]).Cast<T>();
            var idf = Engine.TensorLog(Engine.TensorDivide(Tensor<T>.CreateDefault([terms], NumOps.FromDouble(documents)), df));
            var termWeights = Engine.TensorMultiply(idf, new Tensor<int>(queryCounts, [terms]).Cast<T>());

            // Documents without terms have max count 0 but no matching counts either; divide by 1 instead.
            var maxCounts = new Tensor<int>(index.MaxTermCount.Select(m => Math.Max(m, 1)).ToArray(), [documents, 1]).Cast<T>();
            var tf = Engine.TensorDivide(
                new Tensor<int>(counts, [documents, terms]).Cast<T>(), Engine.TensorBroadcastTo(maxCounts, [documents, terms]));

            return Engine.TensorMatMul(tf, Engine.Reshape(termWeights, [terms, 1])).ToArray();
        }

        internal static List<string> Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }

        private TermIndex GetOrBuildIndex(List<Document<T>> documents)
        {
            lock (_cacheLock)
            {
                // Return cached statistics if the corpus is unchanged
                if (_cachedIndex != null && _cachedIndex.Indexes(documents))
                {
                    return _cachedIndex;
                }

                // Build new statistics and cache them
                var index = BuildIndex(documents);
                _cachedIndex = index;

                return index;
            }
        }

        /// <summary>Counts every document's terms and every term's document frequency.</summary>
        internal static TermIndex BuildIndex(List<Document<T>> documents)
        {
            var index = new TermIndex();
            foreach (var doc in documents)
            {
                var termCounts = new Dictionary<string, int>();
                foreach (var term in Tokenize(doc.Content))
                {
                    termCounts[term] = termCounts.TryGetValue(term, out var count) ? count + 1 : 1;
                }

                foreach (var term in termCounts.Keys)
                {
                    index.DocumentFrequency[term] = index.DocumentFrequency.TryGetValue(term, out var docCount) ? docCount + 1 : 1;
                }

                index.DocumentIds.Add(doc.Id);
                index.Contents.Add(doc.Content);
                index.TermCounts.Add(termCounts);
                index.MaxTermCount.Add(termCounts.Count == 0 ? 0 : termCounts.Values.Max());
            }

            return index;
        }

        /// <summary>Integer term statistics of an indexed corpus.</summary>
        internal sealed class TermIndex
        {
            public List<string> DocumentIds { get; } = new List<string>();
            public List<string> Contents { get; } = new List<string>();
            public List<Dictionary<string, int>> TermCounts { get; } = new List<Dictionary<string, int>>();
            public List<int> MaxTermCount { get; } = new List<int>();
            public Dictionary<string, int> DocumentFrequency { get; } = new Dictionary<string, int>();

            /// <summary>
            /// True when these statistics were built from exactly <paramref name="documents"/>: the same ids
            /// with the same content in the same order. A count check alone kept serving the old statistics
            /// after a document was replaced.
            /// </summary>
            public bool Indexes(List<Document<T>> documents)
            {
                if (documents.Count != DocumentIds.Count)
                    return false;

                for (int i = 0; i < documents.Count; i++)
                {
                    if (!string.Equals(documents[i].Id, DocumentIds[i], StringComparison.Ordinal)
                        || !string.Equals(documents[i].Content, Contents[i], StringComparison.Ordinal))
                        return false;
                }

                return true;
            }
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
