namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedCP (Conditional Policy) personalization with input-dependent routing.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different clients have different types of data. Rather than
/// forcing all data through the same model path, FedCP learns a "routing policy" per client
/// that decides which parts of the model to use for each input. The policy network is lightweight
/// and personalized (kept local), while the main model modules are shared globally. This way,
/// each client can effectively use a different "subset" of the global model tailored to their data.</para>
///
/// <para>Architecture:</para>
/// <code>
/// Input → PolicyNetwork(local) → routing_weights[1..K]
/// Input → Module_1, Module_2, ..., Module_K (shared)
/// Output = sum(routing_weight_k * Module_k(input))
/// </code>
///
/// <para>Reference: Zhang, J., et al. (2023). "Federated Learning with Conditional Computation."
/// KDD 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedCPPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _numExperts;
    private readonly double _policyFraction;

    /// <summary>
    /// Creates a new FedCP personalization strategy.
    /// </summary>
    /// <param name="numExperts">Number of expert modules (K). Default: 4.</param>
    /// <param name="policyFraction">Fraction of total params for the local policy network. Default: 0.05.</param>
    public FedCPPersonalization(int numExperts = 4, double policyFraction = 0.05)
    {
        if (numExperts < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numExperts), "Must have at least 2 experts.");
        }

        if (policyFraction <= 0 || policyFraction >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(policyFraction), "Policy fraction must be in (0, 1).");
        }

        _numExperts = numExperts;
        _policyFraction = policyFraction;
    }

    /// <summary>
    /// Extracts shared expert module parameters for aggregation.
    /// </summary>
    public Dictionary<string, T[]> ExtractSharedParameters(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int policyLayerCount = (int)(layerNames.Length * _policyFraction);
        int sharedCount = layerNames.Length - policyLayerCount;

        var shared = new Dictionary<string, T[]>(sharedCount);
        for (int i = 0; i < sharedCount; i++)
        {
            shared[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return shared;
    }

    /// <summary>
    /// Extracts local policy network parameters (not aggregated).
    /// </summary>
    public Dictionary<string, T[]> ExtractPolicyParameters(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int policyLayerCount = (int)(layerNames.Length * _policyFraction);
        int sharedCount = layerNames.Length - policyLayerCount;

        var policy = new Dictionary<string, T[]>(policyLayerCount);
        for (int i = sharedCount; i < layerNames.Length; i++)
        {
            policy[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return policy;
    }

    /// <summary>
    /// Computes routing weights from the policy network output using softmax.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The policy network takes an input and produces K raw scores
    /// (one per expert module). Softmax converts these into probabilities that sum to 1. The
    /// resulting weights determine how much each expert contributes to the final output.</para>
    /// </remarks>
    /// <param name="policyLogits">Raw logits from the policy network, one per expert.</param>
    /// <returns>Routing weights (softmax probabilities) summing to 1.</returns>
    public double[] ComputeRoutingWeights(T[] policyLogits)
    {
        int k = Math.Min(policyLogits.Length, _numExperts);
        var weights = new double[k];

        double maxLogit = double.NegativeInfinity;
        for (int i = 0; i < k; i++)
        {
            double v = NumOps.ToDouble(policyLogits[i]);
            if (v > maxLogit) maxLogit = v;
        }

        double sumExp = 0;
        for (int i = 0; i < k; i++)
        {
            weights[i] = Math.Exp(NumOps.ToDouble(policyLogits[i]) - maxLogit);
            sumExp += weights[i];
        }

        for (int i = 0; i < k; i++)
        {
            weights[i] /= sumExp;
        }

        return weights;
    }

    /// <summary>
    /// Combines expert outputs using routing weights: output = sum(w_k * expert_k(input)).
    /// </summary>
    /// <param name="expertOutputs">Output from each expert module (K arrays of equal length).</param>
    /// <param name="routingWeights">Routing weights from ComputeRoutingWeights.</param>
    /// <returns>Weighted combination of expert outputs.</returns>
    public T[] CombineExpertOutputs(T[][] expertOutputs, double[] routingWeights)
    {
        if (expertOutputs.Length == 0)
        {
            return [];
        }

        int dim = expertOutputs[0].Length;
        var combined = new T[dim];

        for (int k = 0; k < expertOutputs.Length && k < routingWeights.Length; k++)
        {
            var wT = NumOps.FromDouble(routingWeights[k]);
            for (int i = 0; i < dim; i++)
            {
                combined[i] = NumOps.Add(combined[i], NumOps.Multiply(expertOutputs[k][i], wT));
            }
        }

        return combined;
    }

    /// <summary>
    /// Computes the load-balancing loss to prevent expert collapse (all traffic to one expert).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Without regularization, the policy might learn to always route
    /// everything to one expert (expert collapse). The load-balancing loss penalizes uneven
    /// expert usage by measuring the deviation from uniform routing. This is the same technique
    /// used in Mixture-of-Experts models like GShard and Switch Transformer.</para>
    /// </remarks>
    /// <param name="routingWeightsBatch">Routing weights for each sample in a batch.</param>
    /// <returns>Load-balancing loss value (0 = perfectly balanced).</returns>
    public double ComputeLoadBalancingLoss(double[][] routingWeightsBatch)
    {
        if (routingWeightsBatch.Length == 0)
        {
            return 0;
        }

        int k = routingWeightsBatch[0].Length;
        int batchSize = routingWeightsBatch.Length;
        var avgLoad = new double[k];

        for (int b = 0; b < batchSize; b++)
        {
            for (int e = 0; e < k; e++)
            {
                avgLoad[e] += routingWeightsBatch[b][e];
            }
        }

        for (int e = 0; e < k; e++)
        {
            avgLoad[e] /= batchSize;
        }

        // Coefficient of variation: penalize deviation from uniform (1/K each).
        double uniform = 1.0 / k;
        double loss = 0;
        for (int e = 0; e < k; e++)
        {
            double dev = avgLoad[e] - uniform;
            loss += dev * dev;
        }

        return loss * k; // Scale by K so the loss is comparable across different expert counts.
    }

    /// <summary>Gets the number of expert modules.</summary>
    public int NumExperts => _numExperts;

    /// <summary>Gets the policy network fraction.</summary>
    public double PolicyFraction => _policyFraction;
}
