using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics
{
    /// <summary>
    /// Dot product metric for vector search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class DotProductMetric<T> : ISimilarityMetric<T>
    {
        /// <inheritdoc/>
        public bool HigherIsBetter => true;

        /// <inheritdoc/>
        public T Calculate(Vector<T> v1, Vector<T> v2)
        {
            return v1.DotProduct(v2);
        }
    }
}
