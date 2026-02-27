namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedAA (Federated Adaptive Aggregation) strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard FedAvg, each client's contribution is weighted only
/// by sample count. FedAA learns better aggregation weights by measuring how similar each
/// client's update direction is to the overall update direction. Clients whose updates are
/// more "aligned" with the consensus get higher weight, while outlier updates get lower weight.</para>
///
/// <para>Weight computation:</para>
/// <code>
/// attention_k = softmax(cos_sim(delta_k, delta_avg) / temperature)
/// w_aggregated = sum(attention_k * delta_k)
/// </code>
///
/// <para>Reference: Adaptive Aggregation for Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedAaAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedAaAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="temperature">Softmax temperature for attention weights. Default: 1.0.
    /// Lower values make attention more peaked (favor most-aligned clients).</param>
    public FedAaAggregationStrategy(double temperature = 1.0)
    {
        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive.", nameof(temperature));
        }

        _temperature = temperature;
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
            var single = clientModels.First().Value;
            return single.ToDictionary(kv => kv.Key, kv => (T[])kv.Value.Clone());
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // First pass: compute the naive weighted average to use as the reference direction.
        var naiveAvg = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            naiveAvg[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        foreach (var kvp in clientModels)
        {
            if (!clientWeights.TryGetValue(kvp.Key, out var w))
            {
                throw new ArgumentException($"Missing weight for client {kvp.Key}.", nameof(clientWeights));
            }

            var nw = NumOps.FromDouble(w / totalWeight);
            foreach (var layerName in layerNames)
            {
                if (!kvp.Value.TryGetValue(layerName, out var cp))
                {
                    throw new ArgumentException($"Client {kvp.Key} missing layer '{layerName}'.", nameof(clientModels));
                }

                var avg = naiveAvg[layerName];
                if (cp.Length != avg.Length)
                {
                    throw new ArgumentException(
                        $"Layer '{layerName}' length mismatch for client {kvp.Key}: client={cp.Length}, expected={avg.Length}.",
                        nameof(clientModels));
                }

                for (int i = 0; i < avg.Length; i++)
                {
                    avg[i] = NumOps.Add(avg[i], NumOps.Multiply(cp[i], nw));
                }
            }
        }

        // Second pass: compute cosine similarity of each client vs. naive average across all layers.
        var clientIds = clientModels.Keys.ToList();
        var similarities = new double[clientIds.Count];

        for (int c = 0; c < clientIds.Count; c++)
        {
            var clientModel = clientModels[clientIds[c]];
            double dot = 0, normClient = 0, normAvg = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var ap = naiveAvg[layerName];
                if (cp.Length != ap.Length)
                {
                    throw new ArgumentException(
                        $"Layer '{layerName}' length mismatch for client {clientIds[c]}: client={cp.Length}, expected={ap.Length}.",
                        nameof(clientModels));
                }

                for (int i = 0; i < cp.Length; i++)
                {
                    double cv = NumOps.ToDouble(cp[i]);
                    double av = NumOps.ToDouble(ap[i]);
                    dot += cv * av;
                    normClient += cv * cv;
                    normAvg += av * av;
                }
            }

            double denom = Math.Sqrt(normClient) * Math.Sqrt(normAvg);
            similarities[c] = denom > 0 ? dot / denom : 0.0;
        }

        // Compute softmax attention weights.
        double maxSim = similarities.Max();
        var expSims = new double[clientIds.Count];
        double expSum = 0;
        for (int c = 0; c < clientIds.Count; c++)
        {
            expSims[c] = Math.Exp((similarities[c] - maxSim) / _temperature);
            expSum += expSims[c];
        }

        // Third pass: aggregate with attention weights.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        for (int c = 0; c < clientIds.Count; c++)
        {
            double attentionWeight = expSum > 0 ? expSims[c] / expSum : 1.0 / clientIds.Count;
            var aw = NumOps.FromDouble(attentionWeight);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                if (cp.Length != rp.Length)
                {
                    throw new ArgumentException(
                        $"Layer '{layerName}' length mismatch for client {clientIds[c]}: client={cp.Length}, expected={rp.Length}.",
                        nameof(clientModels));
                }

                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], aw));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the softmax temperature for attention weighting.
    /// </summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedAA(τ={_temperature})";
}
