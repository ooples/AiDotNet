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
    /// Fuses the results of any number of retrievers via Reciprocal Rank Fusion (RRF), with optional
    /// per-retriever weights. Generalizes <see cref="HybridRetriever{T}"/> beyond two retrievers, matching
    /// the LangChain/LlamaIndex "ensemble retriever" (Cormack et al., 2009).
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class EnsembleRetriever<T> : RetrieverBase<T>
    {
        private readonly IReadOnlyList<IRetriever<T>> _retrievers;
        private readonly IReadOnlyList<double> _weights;
        private readonly int _rrfK;

        /// <param name="retrievers">The retrievers to fuse (at least one).</param>
        /// <param name="weights">Optional per-retriever weights (defaults to 1.0 each); length must match.</param>
        /// <param name="defaultTopK">Default number of documents to return.</param>
        /// <param name="rrfK">RRF constant (default 60); larger damps low-rank influence.</param>
        public EnsembleRetriever(
            IReadOnlyList<IRetriever<T>> retrievers,
            IReadOnlyList<double>? weights = null,
            int defaultTopK = 5,
            int rrfK = 60)
            : base(defaultTopK)
        {
            if (retrievers == null) throw new ArgumentNullException(nameof(retrievers));
            if (retrievers.Count == 0) throw new ArgumentException("At least one retriever is required.", nameof(retrievers));
            if (retrievers.Any(r => r == null)) throw new ArgumentException("Retrievers must not be null.", nameof(retrievers));
            if (weights != null && weights.Count != retrievers.Count)
                throw new ArgumentException("weights length must match retrievers length.", nameof(weights));
            if (rrfK <= 0) throw new ArgumentOutOfRangeException(nameof(rrfK), "RRF k must be positive.");

            _retrievers = retrievers;
            _weights = weights ?? Enumerable.Repeat(1.0, retrievers.Count).ToList();
            _rrfK = rrfK;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var rrfScores = new Dictionary<string, double>();
            var docById = new Dictionary<string, Document<T>>();

            for (int r = 0; r < _retrievers.Count; r++)
            {
                var results = _retrievers[r].Retrieve(query, topK * 2, metadataFilters).ToList();
                double weight = _weights[r];
                for (int rank = 0; rank < results.Count; rank++)
                {
                    var doc = results[rank];
                    double contribution = weight / (_rrfK + rank + 1);
                    rrfScores[doc.Id] = rrfScores.TryGetValue(doc.Id, out var s) ? s + contribution : contribution;
                    if (!docById.ContainsKey(doc.Id))
                    {
                        docById[doc.Id] = doc;
                    }
                }
            }

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
