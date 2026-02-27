namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements the FLTrust aggregation strategy for Byzantine-robust federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard federated learning, malicious clients can send
/// fake updates to corrupt the global model. FLTrust solves this by having the server maintain
/// a small, clean "root" dataset. The server computes its own gradient on this data, then
/// scores each client's update by how similar its direction is to the server's gradient.
/// Only client updates that point in roughly the same direction as the server's are included,
/// and they are re-scaled to the server gradient's magnitude to prevent magnitude attacks.</para>
///
/// <para>Trust score computation:</para>
/// <code>
/// ts_k = max(0, cos_sim(g_k, g_server))        // ReLU'd cosine similarity
/// g_k_normalized = ||g_server|| * (g_k / ||g_k||)  // re-scale to server magnitude
/// g_global = sum(ts_k * g_k_normalized) / sum(ts_k)
/// </code>
///
/// <para>Reference: Cao, X., et al. (2021). "FLTrust: Byzantine-robust Federated Learning
/// via Trust Bootstrapping." NDSS 2021.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FLTrustAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private Dictionary<string, T[]>? _serverGradient;

    /// <summary>
    /// Initializes a new instance of the <see cref="FLTrustAggregationStrategy{T}"/> class.
    /// </summary>
    public FLTrustAggregationStrategy()
    {
    }

    /// <summary>
    /// Sets the server's root-dataset gradient used as the trust anchor.
    /// Must be called before <see cref="Aggregate"/> each round.
    /// </summary>
    /// <param name="serverGradient">Parameter dictionary representing the server's gradient.</param>
    public void SetServerGradient(Dictionary<string, T[]> serverGradient)
    {
        _serverGradient = serverGradient ?? throw new ArgumentNullException(nameof(serverGradient));
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientModels.Count == 1)
        {
            return clientModels.First().Value;
        }

        // If no server gradient is set, fall back to weighted average (graceful degradation).
        if (_serverGradient == null)
        {
            return AggregateWeightedAverage(clientModels, clientWeights);
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // Compute server gradient norm.
        double serverNorm = 0;
        foreach (var layerName in layerNames)
        {
            if (!_serverGradient.TryGetValue(layerName, out var sg))
            {
                throw new ArgumentException($"Server gradient missing layer '{layerName}'.");
            }

            for (int i = 0; i < sg.Length; i++)
            {
                double v = NumOps.ToDouble(sg[i]);
                serverNorm += v * v;
            }
        }

        serverNorm = Math.Sqrt(serverNorm);

        // Compute trust scores (ReLU'd cosine similarity with server gradient).
        var clientIds = clientModels.Keys.ToList();
        var trustScores = new double[clientIds.Count];

        for (int c = 0; c < clientIds.Count; c++)
        {
            var clientModel = clientModels[clientIds[c]];
            double dot = 0, clientNorm = 0;

            foreach (var layerName in layerNames)
            {
                if (!clientModel.TryGetValue(layerName, out var cp))
                {
                    throw new ArgumentException($"Client {clientIds[c]} missing layer '{layerName}'.", nameof(clientModels));
                }

                var sg = _serverGradient[layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double cv = NumOps.ToDouble(cp[i]);
                    double sv = NumOps.ToDouble(sg[i]);
                    dot += cv * sv;
                    clientNorm += cv * cv;
                }
            }

            clientNorm = Math.Sqrt(clientNorm);
            double cosSim = (clientNorm > 0 && serverNorm > 0) ? dot / (clientNorm * serverNorm) : 0.0;
            trustScores[c] = Math.Max(0.0, cosSim); // ReLU
        }

        double trustSum = trustScores.Sum();

        // Aggregate: normalize each client update to server magnitude, weight by trust score.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        for (int c = 0; c < clientIds.Count; c++)
        {
            if (trustScores[c] <= 0)
            {
                continue;
            }

            var clientModel = clientModels[clientIds[c]];

            // Compute this client's gradient norm.
            double cNorm = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double v = NumOps.ToDouble(cp[i]);
                    cNorm += v * v;
                }
            }

            cNorm = Math.Sqrt(cNorm);
            double scale = (cNorm > 0 && trustSum > 0)
                ? (serverNorm / cNorm) * (trustScores[c] / trustSum)
                : 0.0;
            var scaleT = NumOps.FromDouble(scale);

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], scaleT));
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override string GetStrategyName() => "FLTrust";
}
