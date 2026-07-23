
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
    /// Hybrid retriever combining dense and sparse retrieval via Reciprocal Rank Fusion (RRF).
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// Dense (cosine, ~0..1) and sparse (BM25, unbounded) scores live on incompatible scales, so a raw
    /// weighted sum is dominated by whichever list has larger-magnitude scores. RRF instead fuses by RANK —
    /// each list contributes <c>weight / (k + rank)</c> — which is scale-free and is the default hybrid
    /// fusion in Elasticsearch/Weaviate/Qdrant and the LangChain/LlamaIndex ensemble retrievers
    /// (Cormack et al., 2009). The weights still let callers favor one retriever; k (default 60) damps the
    /// influence of low ranks.
    /// </remarks>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class HybridRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _denseRetriever;
        private readonly IRetriever<T> _sparseRetriever;
        private readonly double _denseWeight;
        private readonly double _sparseWeight;
        private readonly int _rrfK;

        public HybridRetriever(
            IRetriever<T> denseRetriever,
            IRetriever<T> sparseRetriever,
            double denseWeight = 0.7,
            double sparseWeight = 0.3,
            int defaultTopK = 5,
            int rrfK = 60)
            : base(defaultTopK)
        {
            if (denseRetriever == null)
                throw new ArgumentNullException(nameof(denseRetriever));
            if (sparseRetriever == null)
                throw new ArgumentNullException(nameof(sparseRetriever));
            if (rrfK <= 0)
                throw new ArgumentOutOfRangeException(nameof(rrfK), "RRF k must be positive.");

            _denseRetriever = denseRetriever;
            _sparseRetriever = sparseRetriever;
            _denseWeight = denseWeight;
            _sparseWeight = sparseWeight;
            _rrfK = rrfK;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var denseResults = _denseRetriever.Retrieve(query, topK * 2, metadataFilters).ToList();
            var sparseResults = _sparseRetriever.Retrieve(query, topK * 2, metadataFilters).ToList();

            var rrfScores = new Dictionary<string, double>();
            var docById = new Dictionary<string, Document<T>>();

            void Fuse(List<Document<T>> results, double weight)
            {
                // RRF uses list ORDER, not the raw scores — so scale differences between dense and sparse
                // retrievers no longer distort the fusion.
                for (int rank = 0; rank < results.Count; rank++)
                {
                    var doc = results[rank];
                    double contribution = weight / (_rrfK + rank + 1); // rank is 0-based → 1-based position
                    rrfScores[doc.Id] = rrfScores.TryGetValue(doc.Id, out var s) ? s + contribution : contribution;
                    if (!docById.ContainsKey(doc.Id))
                    {
                        docById[doc.Id] = doc;
                    }
                }
            }

            Fuse(denseResults, _denseWeight);
            Fuse(sparseResults, _sparseWeight);

            return rrfScores
                .OrderByDescending(kv => kv.Value)
                .Take(topK)
                .Select(kv =>
                {
                    var doc = docById[kv.Key];
                    doc.RelevanceScore = NumOps.FromDouble(kv.Value);
                    doc.HasRelevanceScore = true;
                    return doc;
                })
                .ToList();
        }
    }
}
