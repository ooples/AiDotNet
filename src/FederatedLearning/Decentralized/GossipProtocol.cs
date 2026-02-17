namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Gossip Protocol — randomized peer-to-peer model exchange for decentralized FL.
/// </summary>
/// <remarks>
/// <para>
/// In gossip-based aggregation, each node randomly selects K peers per round and averages
/// its model with theirs. Over many rounds, this converges to the global average. The
/// protocol is resilient to node failures since no single node is critical.
/// </para>
/// <para>
/// <b>For Beginners:</b> Like spreading a rumor — each person tells a few random friends,
/// who tell their friends, and eventually everyone knows. Similarly, each device shares its
/// model with a few random peers each round. After enough rounds, all models converge to
/// the same global knowledge.
/// </para>
/// <para>
/// Reference: Boyd et al. (2006), "Randomized Gossip Algorithms".
/// </para>
/// </remarks>
public class GossipProtocol : IDecentralizedTopology
{
    private readonly int _fanout;
    private readonly int _seed;

    /// <inheritdoc/>
    public string TopologyName => $"Gossip(fanout={_fanout})";

    /// <summary>
    /// Creates a new gossip protocol topology.
    /// </summary>
    /// <param name="fanout">Number of random peers to contact per round. Default: 2.</param>
    /// <param name="seed">Random seed for reproducibility. Default: 42.</param>
    public GossipProtocol(int fanout = 2, int seed = 42)
    {
        _fanout = fanout;
        _seed = seed;
    }

    /// <inheritdoc/>
    public int[] GetPeers(int nodeId, int totalNodes, int round)
    {
        if (totalNodes <= 1) return [];

        var rng = new Random(_seed ^ (nodeId * 7919 + round * 6271));
        var peers = new HashSet<int>();

        int maxAttempts = _fanout * 10;
        int attempts = 0;
        while (peers.Count < Math.Min(_fanout, totalNodes - 1) && attempts < maxAttempts)
        {
            int peer = rng.Next(totalNodes);
            if (peer != nodeId)
                peers.Add(peer);
            attempts++;
        }

        return [.. peers];
    }

    /// <inheritdoc/>
    public double GetMixingWeight(int nodeId, int peerId, int totalNodes)
    {
        // Equal weight between self and all peers
        int numPeers = Math.Min(_fanout, totalNodes - 1);
        return 1.0 / (numPeers + 1); // +1 for self
    }
}
