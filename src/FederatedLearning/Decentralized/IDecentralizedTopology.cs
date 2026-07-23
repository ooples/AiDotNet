namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Interface for decentralized peer-to-peer network topologies in serverless federated learning.
/// </summary>
/// <remarks>
/// <para>
/// In decentralized FL, there is no central server. Nodes communicate directly with peers
/// according to a network topology. The topology determines which nodes can exchange models
/// and affects convergence speed and communication costs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of a star pattern (everyone talks to one server), decentralized
/// FL uses patterns like a ring (pass the model around in a circle) or gossip (randomly share
/// with a few neighbors). This removes the single point of failure and can be more robust.
/// </para>
/// </remarks>
public interface IDecentralizedTopology
{
    /// <summary>
    /// Gets the list of peer IDs that a given node should communicate with in this round.
    /// </summary>
    /// <param name="nodeId">The ID of the querying node.</param>
    /// <param name="totalNodes">Total number of nodes in the network.</param>
    /// <param name="round">Current communication round.</param>
    /// <returns>List of peer node IDs to exchange with.</returns>
    int[] GetPeers(int nodeId, int totalNodes, int round);

    /// <summary>
    /// Gets the mixing weight for a peer's contribution during aggregation.
    /// </summary>
    /// <param name="nodeId">The local node ID.</param>
    /// <param name="peerId">The peer node ID.</param>
    /// <param name="totalNodes">Total number of nodes.</param>
    /// <returns>The mixing weight (must be non-negative; all weights for a node sum to 1).</returns>
    double GetMixingWeight(int nodeId, int peerId, int totalNodes);

    /// <summary>
    /// Gets the topology name for logging.
    /// </summary>
    string TopologyName { get; }
}
