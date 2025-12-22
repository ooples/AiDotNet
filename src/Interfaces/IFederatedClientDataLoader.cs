using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a data loader that can provide per-client datasets for federated learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Federated learning uses many small datasets (one per client/device/organization) instead of a single centralized dataset.
/// This interface allows a data loader to expose those natural partitions while still supporting the standard
/// <see cref="IInputOutputDataLoader{T, TInput, TOutput}"/> facade for aggregated access.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "normal data loader" that also knows how to give you
/// each client's local data separately, so federated learning can train realistically.
/// </para>
/// </remarks>
public interface IFederatedClientDataLoader<T, TInput, TOutput> : IInputOutputDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the per-client datasets used for federated learning simulation.
    /// </summary>
    /// <remarks>
    /// Keys are stable client IDs (typically 0..N-1). Values contain each client's local features and labels.
    /// </remarks>
    IReadOnlyDictionary<int, FederatedClientDataset<TInput, TOutput>> ClientData { get; }
}

