namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Implements DFedAvgM — Decentralized FedAvg with Momentum for peer-to-peer FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In decentralized FL, there's no central server. Clients
/// communicate directly with their neighbors in a network graph. DFedAvgM improves on basic
/// decentralized averaging by adding momentum to the averaging step, which smooths out the
/// oscillations caused by heterogeneous data and sparse communication graphs.</para>
///
/// <para>Algorithm per round:</para>
/// <code>
/// 1. Each client trains locally for E epochs
/// 2. Each client averages with neighbors: w_k = sum(mixing_weight_kj * w_j)
/// 3. Apply momentum: m_k = beta * m_k + (1 - beta) * (w_k - w_k_prev)
/// 4. Update: w_k = w_k + m_k
/// </code>
///
/// <para>Reference: Sun, T., et al. (2023). "Decentralized Federated Averaging with Momentum."
/// TMLR 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class DFedAvgMProtocol<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _momentum;
    private Dictionary<int, Dictionary<string, T[]>>? _momentumBuffers;

    /// <summary>
    /// Creates a new DFedAvgM protocol.
    /// </summary>
    /// <param name="momentum">Momentum coefficient (beta). Default: 0.9.</param>
    public DFedAvgMProtocol(double momentum = 0.9)
    {
        if (momentum < 0 || momentum > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1].");
        }

        _momentum = momentum;
    }

    /// <summary>
    /// Performs one round of decentralized averaging with momentum for a client.
    /// </summary>
    /// <param name="clientId">This client's ID.</param>
    /// <param name="currentModel">Client's current model parameters.</param>
    /// <param name="neighborModels">Models from this client's neighbors (includes self).</param>
    /// <param name="mixingWeights">Mixing matrix row for this client (neighbor weights).</param>
    /// <returns>Updated model parameters after averaging + momentum.</returns>
    public Dictionary<string, T[]> AverageWithMomentum(
        int clientId,
        Dictionary<string, T[]> currentModel,
        Dictionary<int, Dictionary<string, T[]>> neighborModels,
        Dictionary<int, double> mixingWeights)
    {
        Guard.NotNull(currentModel);
        Guard.NotNull(neighborModels);
        Guard.NotNull(mixingWeights);
        _momentumBuffers ??= new Dictionary<int, Dictionary<string, T[]>>();

        var layerNames = currentModel.Keys.ToArray();

        // Step 1: Weighted average with neighbors.
        double totalWeight = mixingWeights.Values.Sum();
        if (totalWeight <= 0)
        {
            totalWeight = 1.0; // Avoid division by zero.
        }
        var averaged = new Dictionary<string, T[]>(currentModel.Count);

        foreach (var layerName in layerNames)
        {
            var result = new T[currentModel[layerName].Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Zero;
            }

            foreach (var (neighborId, nw) in mixingWeights)
            {
                if (!neighborModels.TryGetValue(neighborId, out var neighborModel))
                {
                    continue;
                }

                var w = NumOps.FromDouble(nw / totalWeight);
                var np = neighborModel[layerName];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = NumOps.Add(result[i], NumOps.Multiply(np[i], w));
                }
            }

            averaged[layerName] = result;
        }

        // Step 2: Momentum update.
        if (!_momentumBuffers.ContainsKey(clientId))
        {
            _momentumBuffers[clientId] = new Dictionary<string, T[]>();
        }

        var mBuf = _momentumBuffers[clientId];
        var beta = NumOps.FromDouble(_momentum);
        var oneMinusBeta = NumOps.FromDouble(1.0 - _momentum);

        foreach (var layerName in layerNames)
        {
            int layerLen = currentModel[layerName].Length;
            if (!mBuf.TryGetValue(layerName, out var existingMom) || existingMom.Length != layerLen)
            {
                var init = new T[layerLen];
                for (int i = 0; i < init.Length; i++)
                {
                    init[i] = NumOps.Zero;
                }

                mBuf[layerName] = init;
            }

            var m = mBuf[layerName];
            var avg = averaged[layerName];
            var cur = currentModel[layerName];

            for (int i = 0; i < m.Length; i++)
            {
                // m[i] = beta * m[i] + (1-beta) * (avg[i] - cur[i])
                var diff = NumOps.Subtract(avg[i], cur[i]);
                m[i] = NumOps.Add(NumOps.Multiply(beta, m[i]), NumOps.Multiply(oneMinusBeta, diff));
                avg[i] = NumOps.Add(avg[i], m[i]);
            }
        }

        return averaged;
    }

    /// <summary>Gets the momentum coefficient.</summary>
    public double Momentum => _momentum;
}
