using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics
{
    /// <summary>
    /// Euclidean distance metric for vector search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class EuclideanDistanceMetric<T> : ISimilarityMetric<T>
    {
        /// <inheritdoc/>
        public bool HigherIsBetter => false;

        /// <inheritdoc/>
        public T Calculate(Vector<T> v1, Vector<T> v2)
        {
            return StatisticsHelper<T>.EuclideanDistance(v1, v2);
        }
    }
}
