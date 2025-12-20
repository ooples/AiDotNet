using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;


namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Reciprocal Rank Fusion for combining multiple ranking lists.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ReciprocalRankFusion<T> : Rerankers.RerankerBase<T>
    {
        private readonly int _k;

        /// <summary>
        /// Gets a value indicating whether this reranker modifies relevance scores.
        /// </summary>
        public override bool ModifiesScores => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReciprocalRankFusion{T}"/> class.
        /// </summary>
        /// <param name="k">The constant k for reciprocal rank formula (default: 60).</param>
        public ReciprocalRankFusion(int k = 60)
        {
            _k = k > 0 ? k : throw new ArgumentOutOfRangeException(nameof(k));
        }

        /// <summary>
        /// Reranks documents using reciprocal rank fusion.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="documents">The documents to rerank.</param>
        /// <returns>A reranked list of documents.</returns>
        protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
        {
            var scores = new Dictionary<string, T>();

            for (int rank = 0; rank < documents.Count; rank++)
            {
                var doc = documents[rank];
                var rrfScore = NumOps.FromDouble(1.0 / (_k + rank + 1));

                scores[doc.Id] = scores.TryGetValue(doc.Id, out var existingScore)
                    ? NumOps.Add(existingScore, rrfScore)
                    : rrfScore;
            }

            var reranked = documents
                .OrderByDescending(d => scores.TryGetValue(d.Id, out var score) ? Convert.ToDouble(score) : 0.0)
                .ToList();

            foreach (var doc in reranked.Where(d => scores.ContainsKey(d.Id)))
            {
                doc.RelevanceScore = scores[doc.Id];
                doc.HasRelevanceScore = true;
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
                    var rrfScore = NumOps.FromDouble(1.0 / (_k + rank + 1));

                    if (scores.TryGetValue(doc.Id, out var existingScore))
                    {
                        scores[doc.Id] = NumOps.Add(existingScore, rrfScore);
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
