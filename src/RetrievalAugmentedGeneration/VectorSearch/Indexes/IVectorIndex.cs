using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Interface for vector search indexes.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [AiDotNet.Configuration.YamlConfigurable("VectorIndex")]
    public interface IVectorIndex<T>
    {
        /// <summary>
        /// Adds a vector to the index.
        /// </summary>
        /// <param name="id">The unique identifier for the vector.</param>
        /// <param name="vector">The vector to add.</param>
        void Add(string id, Vector<T> vector);

        /// <summary>
        /// Adds multiple vectors to the index in batch.
        /// </summary>
        /// <param name="vectors">Dictionary of id to vector mappings.</param>
        void AddBatch(Dictionary<string, Vector<T>> vectors);

        /// <summary>
        /// Searches for the k nearest neighbors to the query vector.
        /// </summary>
        /// <param name="query">The query vector.</param>
        /// <param name="k">The number of neighbors to return.</param>
        /// <returns>List of (id, score) tuples ordered by similarity.</returns>
        List<(string Id, T Score)> Search(Vector<T> query, int k);

        /// <summary>
        /// Removes a vector from the index.
        /// </summary>
        /// <param name="id">The id of the vector to remove.</param>
        /// <returns>True if the vector was removed, false if not found.</returns>
        bool Remove(string id);

        /// <summary>
        /// Gets the number of vectors in the index.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Clears all vectors from the index.
        /// </summary>
        void Clear();
    }
}
