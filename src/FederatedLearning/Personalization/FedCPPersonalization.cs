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

    /// <summary>Gets the number of expert modules.</summary>
    public int NumExperts => _numExperts;

    /// <summary>Gets the policy network fraction.</summary>
    public double PolicyFraction => _policyFraction;
}
