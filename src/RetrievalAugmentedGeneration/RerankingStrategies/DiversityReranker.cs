using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Diversity-based reranking to maximize coverage
    /// </summary>
    /// <typeparam name="T">The numeric type for scoring</typeparam>
    public class DiversityReranker<T> : RerankingStrategyBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly T _diversityWeight;

        public DiversityReranker(T diversityWeight = default)
        {
            _diversityWeight = NumOps.Equals(diversityWeight, NumOps.Zero)
                ? NumOps.FromDouble(0.5)
                : diversityWeight;
        }

        protected override Task<List<Document<T>>> RerankCoreAsync(string query, List<Document<T>> documents, int topK)
        {
            if (documents == null || documents.Count == 0)
                return Task.FromResult(new List<Document<T>>());

            var selected = new List<Document<T>>();
            var remaining = new List<Document<T>>(documents);

            selected.Add(remaining[0]);
            remaining.RemoveAt(0);

            while (selected.Count < topK && remaining.Count > 0)
            {
                var bestDoc = remaining[0];
                var bestScore = NumOps.FromDouble(-1.0);

                foreach (var doc in remaining)
                {
                    var minSimilarity = NumOps.One;
                    foreach (var selectedDoc in selected)
                    {
                        var similarity = StatisticsHelper.CosineSimilarity(
                            doc.Embedding,
                            selectedDoc.Embedding,
                            NumOps);

                        if (NumOps.LessThan(similarity, minSimilarity))
                        {
                            minSimilarity = similarity;
                        }
                    }

                    var diversityScore = NumOps.Subtract(NumOps.One, minSimilarity);
                    var combinedScore = NumOps.Multiply(diversityScore, _diversityWeight);

                    if (NumOps.GreaterThan(combinedScore, bestScore))
                    {
                        bestScore = combinedScore;
                        bestDoc = doc;
                    }
                }

                selected.Add(bestDoc);
                remaining.Remove(bestDoc);
            }

            return Task.FromResult(selected);
        }
    }
}
