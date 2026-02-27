namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Implements time-varying topology for decentralized federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In decentralized FL, the communication graph determines which
/// clients can talk to each other. A fixed graph can create "bottleneck" nodes and slow convergence.
/// Time-varying topology changes the graph each round — this accelerates mixing (spreading
/// information across all clients) and makes the system more robust to node failures.</para>
///
/// <para>Strategies:</para>
/// <list type="bullet">
/// <item><b>RandomPairing</b> — Each round, randomly pair clients for gossip</item>
/// <item><b>CyclicPermutation</b> — Rotate the communication pattern deterministically</item>
/// <item><b>Exponential</b> — Each client connects to client at distance 2^k for k=0,1,...</item>
/// </list>
///
/// <para>Reference: Time-Varying Communication Topologies for Decentralized FL (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class TimeVaryingTopology<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    /// <summary>Topology generation strategy.</summary>
    public enum TopologyStrategy
    {
        /// <summary>Random pairing each round.</summary>
        RandomPairing,
        /// <summary>Cyclic shift of neighbors.</summary>
        CyclicPermutation,
        /// <summary>Exponential graph (connect to node at distance 2^k).</summary>
        Exponential
    }

    private readonly TopologyStrategy _strategy;
    private readonly int _seed;
    private int _roundCounter;

    /// <summary>
    /// Creates a new time-varying topology.
    /// </summary>
    /// <param name="strategy">Topology generation strategy. Default: RandomPairing.</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    public TimeVaryingTopology(TopologyStrategy strategy = TopologyStrategy.RandomPairing, int seed = 42)
    {
        _strategy = strategy;
        _seed = seed;
    }

    /// <summary>
    /// Generates the neighbor set for each client for the current round.
    /// </summary>
    /// <param name="clientIds">All client IDs in the system.</param>
    /// <returns>Dictionary of clientId to their neighbor set for this round.</returns>
    public Dictionary<int, List<int>> GenerateTopology(IReadOnlyList<int> clientIds)
    {
        var topology = new Dictionary<int, List<int>>();
        foreach (var id in clientIds)
        {
            topology[id] = new List<int> { id }; // Always include self.
        }

        int n = clientIds.Count;
        if (n <= 1)
        {
            _roundCounter++;
            return topology;
        }

        switch (_strategy)
        {
            case TopologyStrategy.RandomPairing:
            {
                var rng = new Random(_seed + _roundCounter);
                var shuffled = clientIds.OrderBy(_ => rng.Next()).ToList();
                for (int i = 0; i < shuffled.Count - 1; i += 2)
                {
                    topology[shuffled[i]].Add(shuffled[i + 1]);
                    topology[shuffled[i + 1]].Add(shuffled[i]);
                }

                break;
            }
            case TopologyStrategy.CyclicPermutation:
            {
                int shift = (_roundCounter % (n - 1)) + 1;
                for (int i = 0; i < n; i++)
                {
                    int neighbor = (i + shift) % n;
                    topology[clientIds[i]].Add(clientIds[neighbor]);
                    topology[clientIds[neighbor]].Add(clientIds[i]);
                }

                break;
            }
            case TopologyStrategy.Exponential:
            {
                for (int i = 0; i < n; i++)
                {
                    for (int k = 0; (1 << k) < n; k++)
                    {
                        int neighbor = (i + (1 << k)) % n;
                        if (!topology[clientIds[i]].Contains(clientIds[neighbor]))
                        {
                            topology[clientIds[i]].Add(clientIds[neighbor]);
                        }
                    }
                }

                break;
            }
        }

        _roundCounter++;
        return topology;
    }

    /// <summary>Gets the topology strategy.</summary>
    public TopologyStrategy Strategy => _strategy;

    /// <summary>Gets the current round.</summary>
    public int RoundCounter => _roundCounter;
}
