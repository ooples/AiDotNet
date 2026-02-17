namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Handles secure discovery and management of edges that cross client boundaries.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a graph is split across clients, some edges connect nodes on
/// different clients (cross-client edges). Discovering these edges is essential for GNN quality
/// but must be done privately — Client A shouldn't learn Client B's full adjacency list.</para>
///
/// <para><b>Approach:</b> Uses Private Set Intersection (PSI) from issue #538: each client provides
/// its border node IDs, and the PSI protocol reveals only the intersection (shared nodes) without
/// exposing non-shared nodes.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ICrossClientEdgeHandler<T>
{
    /// <summary>
    /// Discovers cross-client edges between two clients using PSI.
    /// </summary>
    /// <param name="clientABorderNodes">Border node IDs from client A.</param>
    /// <param name="clientBBorderNodes">Border node IDs from client B.</param>
    /// <returns>Discovered cross-client edges as (nodeA, nodeB) pairs.</returns>
    IReadOnlyList<(int NodeA, int NodeB)> DiscoverEdges(
        IReadOnlyList<int> clientABorderNodes,
        IReadOnlyList<int> clientBBorderNodes);

    /// <summary>
    /// Gets the total number of discovered cross-client edges.
    /// </summary>
    int DiscoveredEdgeCount { get; }

    /// <summary>
    /// Gets the discovered cross-client edges between a specific client pair.
    /// </summary>
    /// <param name="clientA">First client ID.</param>
    /// <param name="clientB">Second client ID.</param>
    /// <returns>Cross-client edges for this pair, or empty if not yet discovered.</returns>
    IReadOnlyList<(int NodeA, int NodeB)> GetEdges(int clientA, int clientB);

    /// <summary>
    /// Caches discovered edges for a client pair (to avoid re-running PSI each round).
    /// </summary>
    /// <param name="clientA">First client ID.</param>
    /// <param name="clientB">Second client ID.</param>
    /// <param name="edges">Discovered edges to cache.</param>
    void CacheEdges(int clientA, int clientB, IReadOnlyList<(int NodeA, int NodeB)> edges);
}
