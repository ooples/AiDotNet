using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics
{
    /// <summary>
    /// Jaccard similarity metric for vector search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class JaccardSimilarityMetric<T> : ISimilarityMetric<T>
    {
        /// <inheritdoc/>
        public bool HigherIsBetter => true;

        /// <inheritdoc/>
        public T Calculate(Vector<T> v1, Vector<T> v2)
        {
            return StatisticsHelper<T>.JaccardSimilarity(v1, v2);
        }
    }
}
