using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// Defines a cache that stores previously computed embedding vectors keyed by a stable string key.
/// </summary>
/// <remarks>
/// <para>
/// A cache lets a <see cref="CacheBackedEmbeddings{T}"/> decorator avoid re-embedding text that has
/// already been embedded. Keys are expected to be content-derived (see <see cref="ContentHash"/>) and
/// namespaced by the underlying model's identity so that different models never share entries.
/// </para>
/// <para><b>For Beginners:</b> This is a lookup table from "fingerprint of some text" to "its embedding".
///
/// - Before asking the (possibly slow or paid) model to embed text, we check the cache.
/// - If the answer is already there, we reuse it instead of computing it again.
/// - Implementations must be safe to call from multiple threads at once.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public interface IEmbeddingCache<T>
{
    /// <summary>
    /// Attempts to retrieve a cached embedding for the given key.
    /// </summary>
    /// <param name="key">The cache key.</param>
    /// <param name="embedding">When this method returns <c>true</c>, contains the cached embedding; otherwise <c>null</c>.</param>
    /// <returns><c>true</c> if a cached embedding was found; otherwise <c>false</c>.</returns>
    bool TryGet(string key, out Vector<T>? embedding);

    /// <summary>
    /// Stores an embedding in the cache under the given key, overwriting any existing entry.
    /// </summary>
    /// <param name="key">The cache key.</param>
    /// <param name="embedding">The embedding to store.</param>
    void Set(string key, Vector<T> embedding);

    /// <summary>
    /// Gets the number of entries currently held in the cache.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Removes all entries from the cache.
    /// </summary>
    void Clear();
}
