using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Reciprocal Rank Fusion for combining multiple ranking lists.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ReciprocalRankFusion<T> : RerankingStrategyBase<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly int _k;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReciprocalRankFusion{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="k">The constant k for reciprocal rank formula (default: 60).</param>
        public ReciprocalRankFusion(INumericOperations<T> numericOperations, int k = 60) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _k = k > 0 ? k : throw new ArgumentOutOfRangeException(nameof(k));
        }

        /// <summary>
        /// Reranks documents using reciprocal rank fusion.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="documents">The documents to rerank.</param>
        /// <param name="topK">The number of top documents to return.</param>
        /// <returns>A reranked list of documents.</returns>
        public override List<Document<T>> Rerank(string query, List<Document<T>> documents, int topK)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (documents == null) throw new ArgumentNullException(nameof(documents));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var scores = new Dictionary<string, T>();

            for (int rank = 0; rank < documents.Count; rank++)
            {
                var doc = documents[rank];
                var rrfScore = _numOps.FromDouble(1.0 / (_k + rank + 1));

                if (scores.ContainsKey(doc.Id))
                {
                    scores[doc.Id] = _numOps.Add(scores[doc.Id], rrfScore);
                }
                else
                {
                    scores[doc.Id] = rrfScore;
                }
            }

            var reranked = documents
                .OrderByDescending(d => Convert.ToDouble(scores.ContainsKey(d.Id) ? scores[d.Id] : _numOps.Zero))
                .Take(topK)
                .ToList();

            foreach (var doc in reranked)
            {
                if (scores.ContainsKey(doc.Id))
                {
                    doc.RelevanceScore = scores[doc.Id];
                    doc.HasRelevanceScore = true;
                }
            }

            return reranked;
        }

        /// <summary>
        /// Fuses multiple ranking lists using reciprocal rank fusion.
        /// </summary>
        /// <param name="rankingLists">Multiple lists of ranked documents.</param>
        /// <param name="topK">The number of top documents to return.</param>
        /// <returns>A fused and reranked list of documents.</returns>
        public List<Document<T>> FuseRankings(List<List<Document<T>>> rankingLists, int topK)
        {
            if (rankingLists == null || rankingLists.Count == 0)
                throw new ArgumentException("Ranking lists cannot be null or empty", nameof(rankingLists));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var scores = new Dictionary<string, T>();
            var allDocs = new Dictionary<string, Document<T>>();

            foreach (var rankingList in rankingLists)
            {
                for (int rank = 0; rank < rankingList.Count; rank++)
                {
                    var doc = rankingList[rank];
                    var rrfScore = _numOps.FromDouble(1.0 / (_k + rank + 1));

                    if (scores.ContainsKey(doc.Id))
                    {
                        scores[doc.Id] = _numOps.Add(scores[doc.Id], rrfScore);
                    }
                    else
                    {
                        scores[doc.Id] = rrfScore;
                        allDocs[doc.Id] = doc;
                    }
                }
            }

            var reranked = allDocs.Values
                .OrderByDescending(d => Convert.ToDouble(scores[d.Id]))
                .Take(topK)
                .ToList();

            foreach (var doc in reranked)
            {
                doc.RelevanceScore = scores[doc.Id];
            }

            return reranked;
        }
    }
}
