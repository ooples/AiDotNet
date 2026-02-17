namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Ring AllReduce — communication-efficient decentralized averaging using a ring topology.
/// </summary>
/// <remarks>
/// <para>
/// In Ring AllReduce, nodes are arranged in a logical ring. Each round consists of two phases:
/// (1) scatter-reduce: each node sends a chunk of its data to the next node in the ring and
/// receives a chunk from the previous node, reducing as it goes; (2) allgather: the fully
/// reduced chunks are propagated around the ring. This achieves bandwidth-optimal communication
/// cost of 2(N-1)/N × data_size regardless of the number of nodes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine passing cards around a circle. Each person adds their card
/// to the pile as it passes through them. After going around twice, everyone has seen all
/// the cards. This is much more efficient than having everyone send their cards to one
/// central person.
/// </para>
/// </remarks>
public class RingAllReduceProtocol : IDecentralizedTopology
{
    /// <inheritdoc/>
    public string TopologyName => "RingAllReduce";

    /// <inheritdoc/>
    public int[] GetPeers(int nodeId, int totalNodes, int round)
    {
        if (totalNodes <= 1) return [];

        // In a ring, each node communicates with its left and right neighbors
        int left = (nodeId - 1 + totalNodes) % totalNodes;
        int right = (nodeId + 1) % totalNodes;

        if (totalNodes == 2)
            return [left == nodeId ? right : left];

        return [left, right];
    }

    /// <inheritdoc/>
    public double GetMixingWeight(int nodeId, int peerId, int totalNodes)
    {
        // Doubly stochastic mixing: self-weight = 1 - degree * w, peer-weight = w
        // For ring topology with degree 2: w = 1/3 each (self + 2 neighbors)
        if (totalNodes <= 2)
            return 0.5;

        return 1.0 / 3.0;
    }
}
