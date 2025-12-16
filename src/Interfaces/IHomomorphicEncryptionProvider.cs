using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Interfaces;

/// <summary>
/// Provides homomorphic encryption operations for federated learning aggregation.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The provider hides cryptographic details (keys, ciphertexts, parameters) behind a simple interface.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public interface IHomomorphicEncryptionProvider<T>
{
    /// <summary>
    /// Aggregates selected parameter indices using homomorphic encryption to produce a weighted average.
    /// </summary>
    /// <param name="clientParameters">Per-client parameter vectors.</param>
    /// <param name="clientWeights">Per-client weights.</param>
    /// <param name="globalBaseline">Baseline vector used for indices not included in <paramref name="encryptedIndices"/>.</param>
    /// <param name="encryptedIndices">Indices to aggregate via HE.</param>
    /// <param name="options">HE options.</param>
    /// <returns>A vector containing the weighted average for encrypted indices (baseline elsewhere).</returns>
    Vector<T> AggregateEncryptedWeightedAverage(
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        Vector<T> globalBaseline,
        IReadOnlyList<int> encryptedIndices,
        HomomorphicEncryptionOptions options);

    /// <summary>
    /// Gets the provider name.
    /// </summary>
    string GetProviderName();
}

