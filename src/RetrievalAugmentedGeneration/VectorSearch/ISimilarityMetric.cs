using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch
{
    /// <summary>
    /// Interface for similarity/distance metrics used in vector search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [AiDotNet.Configuration.YamlConfigurable("SimilarityMetric")]
    public interface ISimilarityMetric<T>
    {
        /// <summary>
        /// Calculates the similarity or distance between two vectors.
        /// </summary>
        /// <param name="v1">The first vector.</param>
        /// <param name="v2">The second vector.</param>
        /// <returns>The similarity or distance value.</returns>
        T Calculate(Vector<T> v1, Vector<T> v2);

        /// <summary>
        /// Gets whether higher values indicate greater similarity (true) or greater distance (false).
        /// </summary>
        bool HigherIsBetter { get; }
    }
}
