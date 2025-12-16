using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Selects which clients participate in a federated learning round.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In federated learning, the server usually does not use every client in every round.
/// Instead, it picks a subset of clients to reduce communication cost and handle device availability.
///
/// Different selection strategies optimize for different goals:
/// - Uniform: simple and fair
/// - Weighted: prefer clients with more data
/// - Stratified: ensure each group is represented
/// - Availability-aware: prefer clients likely to be online
/// - Performance-aware: prefer clients that historically help training
/// - Cluster-based: sample across diverse client behaviors
/// </remarks>
public interface IClientSelectionStrategy
{
    /// <summary>
    /// Selects the clients to participate for the current round.
    /// </summary>
    /// <param name="request">The selection request containing available clients and selection context.</param>
    /// <returns>The selected client IDs.</returns>
    List<int> SelectClients(ClientSelectionRequest request);

    /// <summary>
    /// Gets the name of the selection strategy.
    /// </summary>
    string GetStrategyName();
}

