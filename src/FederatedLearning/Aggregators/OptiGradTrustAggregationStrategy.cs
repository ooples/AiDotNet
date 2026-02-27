namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements OptiGradTrust (Optimized Gradient Trust) aggregation strategy with
/// historical reputation tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> OptiGradTrust builds on trust-based defenses like FLTrust by
/// adding a historical reputation system. Each client maintains a trust score that is updated
/// over multiple rounds. A client that consistently sends aligned, constructive updates builds
/// a higher reputation, while one that repeatedly deviates gets downweighted. This makes the
/// defense more resilient to adaptive attackers who behave honestly for a few rounds then
/// suddenly attack.</para>
///
/// <para>Trust update rule:</para>
/// <code>
/// current_trust_k = max(0, cos_sim(g_k, g_mean))
/// reputation_k = momentum * reputation_k + (1 - momentum) * current_trust_k
/// w_k = reputation_k / sum(reputation_j)
/// </code>
///
/// <para>Reference: Optimized Gradient Trust Scoring for Federated Learning (2025).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class OptiGradTrustAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _momentum;
    private readonly double _minReputation;
    private Dictionary<int, double>? _reputations;

    /// <summary>
    /// Initializes a new instance of the <see cref="OptiGradTrustAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="momentum">EMA momentum for reputation updates. Higher values weigh history more.
    /// Default: 0.9.</param>
    /// <param name="minReputation">Floor for reputation scores to prevent permanent exclusion.
    /// Default: 0.01.</param>
    public OptiGradTrustAggregationStrategy(double momentum = 0.9, double minReputation = 0.01)
    {
        if (momentum < 0 || momentum > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1].");
        }

        if (minReputation < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minReputation), "Minimum reputation must be non-negative.");
        }

        _momentum = momentum;
        _minReputation = minReputation;
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

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        // Initialize reputations for new clients.
        _reputations ??= new Dictionary<int, double>();
        foreach (var clientId in clientIds)
        {
            if (!_reputations.ContainsKey(clientId))
            {
                _reputations[clientId] = 1.0; // Start with full trust.
            }
        }

        // Compute the weighted mean update as reference direction.
        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var meanFlat = new double[totalParams];

        for (int c = 0; c < n; c++)
        {
            double w = clientWeights.TryGetValue(clientIds[c], out var cw) ? cw / totalWeight : 1.0 / n;
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    meanFlat[offset++] += NumOps.ToDouble(cp[i]) * w;
                }
            }
        }

        double meanNorm = 0;
        for (int i = 0; i < totalParams; i++)
        {
            meanNorm += meanFlat[i] * meanFlat[i];
        }

        meanNorm = Math.Sqrt(meanNorm);

        // Compute current trust scores and update reputations.
        var currentTrust = new double[n];
        for (int c = 0; c < n; c++)
        {
            double dot = 0, clientNorm = 0;
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double cv = NumOps.ToDouble(cp[i]);
                    dot += cv * meanFlat[offset];
                    clientNorm += cv * cv;
                    offset++;
                }
            }

            clientNorm = Math.Sqrt(clientNorm);
            double cosSim = (clientNorm > 0 && meanNorm > 0) ? dot / (clientNorm * meanNorm) : 0.0;
            currentTrust[c] = Math.Max(0.0, cosSim);
        }

        // EMA reputation update.
        for (int c = 0; c < n; c++)
        {
            double oldRep = _reputations[clientIds[c]];
            double newRep = _momentum * oldRep + (1.0 - _momentum) * currentTrust[c];
            _reputations[clientIds[c]] = Math.Max(_minReputation, newRep);
        }

        // Aggregate with reputation-based weights.
        double repSum = clientIds.Sum(id => _reputations[id]);

        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        for (int c = 0; c < n; c++)
        {
            double repWeight = repSum > 0 ? _reputations[clientIds[c]] / repSum : 1.0 / n;
            var rw = NumOps.FromDouble(repWeight);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], rw));
                }
            }
        }

        return result;
    }

    /// <summary>Gets the EMA momentum for reputation updates.</summary>
    public double Momentum => _momentum;

    /// <summary>Gets the minimum reputation floor.</summary>
    public double MinReputation => _minReputation;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"OptiGradTrust(μ={_momentum})";
}
