using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.Models;


namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Reciprocal Rank Fusion for combining multiple ranking lists.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Reranker)]
    [PipelineStage(PipelineStage.PostRetrieval)]
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
            var (index, scores) = ReciprocalRankScores(new[] { documents });

            var reranked = documents
                .OrderByDescending(d => scores[index[d.Id]])
                .ToList();

            foreach (var doc in reranked)
            {
                doc.RelevanceScore = scores[index[doc.Id]];
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

            var allDocs = new Dictionary<string, Document<T>>();
            foreach (var doc in rankingLists.SelectMany(list => list))
            {
                if (!allDocs.ContainsKey(doc.Id))
                    allDocs[doc.Id] = doc;
            }

            var (index, scores) = ReciprocalRankScores(rankingLists);

            var reranked = allDocs.Values
                .OrderByDescending(d => scores[index[d.Id]])
                .Take(topK)
                .ToList();

            foreach (var doc in reranked)
            {
                doc.RelevanceScore = scores[index[doc.Id]];
                doc.HasRelevanceScore = true;
            }

            return reranked;
        }

        /// <summary>
        /// RRF(d) = Σ_lists 1 / (k + r(d)) with r the 1-based rank (Cormack, Clarke and Buettcher 2009).
        /// </summary>
        /// <returns>Each document id's position in the score array, and the fused scores.</returns>
        private (Dictionary<string, int> Index, T[] Scores) ReciprocalRankScores(IEnumerable<IList<Document<T>>> rankingLists)
        {
            var index = new Dictionary<string, int>();
            var occurrences = new List<int>();
            var ranks = new List<int>();
            foreach (var rankingList in rankingLists)
            {
                for (int rank = 0; rank < rankingList.Count; rank++)
                {
                    var id = rankingList[rank].Id;
                    if (!index.TryGetValue(id, out var position))
                    {
                        position = index.Count;
                        index[id] = position;
                    }

                    occurrences.Add(position);
                    ranks.Add(rank + 1);
                }
            }

            int m = ranks.Count;
            if (m == 0)
                return (index, Array.Empty<T>());

            var denominators = Engine.TensorAddScalar(new Tensor<int>(ranks.ToArray(), [m]).Cast<T>(), NumOps.FromDouble(_k));
            var contributions = Engine.TensorDivide(Tensor<T>.CreateDefault([m], NumOps.One), denominators);
            var membership = Engine.TensorOneHot<T>(new Tensor<int>(occurrences.ToArray(), [m]), index.Count);
            var totals = Engine.TensorMatMul(Engine.Reshape(contributions, [1, m]), membership);
            return (index, totals.ToArray());
        }
    }
}
