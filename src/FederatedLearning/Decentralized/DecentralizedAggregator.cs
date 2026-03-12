namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Decentralized aggregator — performs local mixing of model parameters based on topology.
/// </summary>
/// <remarks>
/// <para>
/// The decentralized aggregator replaces the central server aggregation step. Each node
/// performs a weighted average of its own model with models received from its peers according
/// to the topology's mixing weights. Over rounds, this converges to the global average.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of sending your model to a central server and getting back
/// an average, you directly blend your model with your neighbors' models. The blending weights
/// come from the topology (e.g., gossip, ring). After enough blending rounds, everyone's
/// model converges to the same global model.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DecentralizedAggregator<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly IDecentralizedTopology _topology;

    /// <summary>
    /// Creates a new decentralized aggregator with the specified topology.
    /// </summary>
    /// <param name="topology">The P2P network topology to use.</param>
    public DecentralizedAggregator(IDecentralizedTopology topology)
    {
        _topology = topology ?? throw new ArgumentNullException(nameof(topology));
    }

    /// <summary>
    /// Gets the topology used by this aggregator.
    /// </summary>
    public IDecentralizedTopology Topology => _topology;

    /// <summary>
    /// Performs one round of decentralized averaging for a single node.
    /// </summary>
    /// <param name="nodeId">The ID of the local node.</param>
    /// <param name="localModel">The local node's model parameters.</param>
    /// <param name="peerModels">Models received from peers (keyed by peer ID).</param>
    /// <param name="totalNodes">Total number of nodes in the network.</param>
    /// <param name="round">Current communication round.</param>
    /// <returns>The locally mixed model parameters.</returns>
    public Vector<T> MixWithPeers(int nodeId, Vector<T> localModel,
        Dictionary<int, Vector<T>> peerModels, int totalNodes, int round)
    {
        int d = localModel.Length;
        var mixed = new T[d];

        // Self-weight
        int[] expectedPeers = _topology.GetPeers(nodeId, totalNodes, round);
        double selfWeight = _topology.GetMixingWeight(nodeId, nodeId, totalNodes);

        // Adjust self-weight so total sums to 1
        double peerWeightSum = 0;
        foreach (int peerId in expectedPeers)
        {
            if (peerModels.ContainsKey(peerId))
            {
                peerWeightSum += _topology.GetMixingWeight(nodeId, peerId, totalNodes);
            }
        }

        // If no peers responded, keep local model
        if (peerWeightSum == 0)
            return localModel;

        double totalWeightUsed = selfWeight + peerWeightSum;
        double normalizedSelfWeight = selfWeight / totalWeightUsed;

        // Apply self-contribution
        T selfW = NumOps.FromDouble(normalizedSelfWeight);
        for (int i = 0; i < d; i++)
        {
            mixed[i] = NumOps.Multiply(localModel[i], selfW);
        }

        // Apply peer contributions
        foreach (int peerId in expectedPeers)
        {
            if (!peerModels.TryGetValue(peerId, out var peerModel))
                continue;

            double peerWeight = _topology.GetMixingWeight(nodeId, peerId, totalNodes) / totalWeightUsed;
            T pw = NumOps.FromDouble(peerWeight);

            int peerLen = Math.Min(d, peerModel.Length);
            for (int i = 0; i < peerLen; i++)
            {
                mixed[i] = NumOps.Add(mixed[i], NumOps.Multiply(peerModel[i], pw));
            }
        }

        return new Vector<T>(mixed);
    }

    /// <summary>
    /// Simulates a full decentralized round where all nodes exchange and mix simultaneously.
    /// </summary>
    /// <param name="nodeModels">All node models keyed by node ID.</param>
    /// <param name="round">Current communication round.</param>
    /// <returns>Updated models for all nodes after one mixing round.</returns>
    public Dictionary<int, Vector<T>> SimulateRound(Dictionary<int, Vector<T>> nodeModels, int round)
    {
        int totalNodes = nodeModels.Count;
        var updated = new Dictionary<int, Vector<T>>();

        foreach (var (nodeId, localModel) in nodeModels)
        {
            int[] peers = _topology.GetPeers(nodeId, totalNodes, round);
            var peerModels = new Dictionary<int, Vector<T>>();
            foreach (int peerId in peers)
            {
                if (nodeModels.TryGetValue(peerId, out var peerModel))
                {
                    peerModels[peerId] = peerModel;
                }
            }

            updated[nodeId] = MixWithPeers(nodeId, localModel, peerModels, totalNodes, round);
        }

        return updated;
    }
}
