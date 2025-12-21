namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements the Federated Proximal (FedProx) aggregation strategy.
/// </summary>
/// <remarks>
/// FedProx is an extension of FedAvg that handles system and statistical heterogeneity
/// in federated learning. It was proposed by Li et al. in 2020 to address challenges
/// when clients have different computational capabilities or data distributions.
///
/// <b>For Beginners:</b> FedProx is like FedAvg with a "safety rope" that prevents
/// individual clients from pulling the shared model too far in their own direction.
///
/// Key differences from FedAvg:
/// 1. Adds a proximal term to local training objective
/// 2. Prevents client models from deviating too much from global model
/// 3. Improves convergence when clients have heterogeneous data or capabilities
///
/// How FedProx works:
/// During local training, each client minimizes:
///   Local Loss + (μ/2) × ||w - w_global||²
///
/// where:
/// - Local Loss: Standard loss on client's data
/// - μ (mu): Proximal term coefficient (controls constraint strength)
/// - w: Client's current model weights
/// - w_global: Global model weights received from server
/// - ||w - w_global||²: Squared distance between client and global model
///
/// For example, with μ = 0.01:
/// - Client trains on local data
/// - Proximal term penalizes large deviations from global model
/// - If client's data is very different, can still adapt but with limitation
/// - Prevents overfitting to local data distribution
///
/// When to use FedProx:
/// - Non-IID data (different distributions across clients)
/// - System heterogeneity (some clients much slower/faster)
/// - Want more stable convergence than FedAvg
/// - Stragglers problem (some clients take much longer)
///
/// Benefits:
/// - Better convergence on non-IID data
/// - More robust to stragglers
/// - Theoretically proven convergence guarantees
/// - Small computational overhead
///
/// Limitations:
/// - Requires tuning μ parameter
/// - Slightly slower local training per iteration
/// - May converge slower if μ is too large
///
/// Reference: Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks."
/// MLSys 2020.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class FedProxAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _mu;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedProxAggregationStrategy{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a FedProx aggregator with a specified proximal term strength.
    ///
    /// The μ (mu) parameter controls the trade-off between local adaptation and global consistency:
    /// - μ = 0: Equivalent to FedAvg (no constraint)
    /// - μ = 0.01: Mild constraint (recommended starting point)
    /// - μ = 0.1: Moderate constraint
    /// - μ = 1.0+: Strong constraint (may be too restrictive)
    ///
    /// Recommendations:
    /// - Start with μ = 0.01
    /// - Increase if convergence is unstable
    /// - Decrease if convergence is too slow
    /// </remarks>
    /// <param name="mu">The proximal term coefficient (typically 0.01 to 1.0).</param>
    public FedProxAggregationStrategy(double mu = 0.01)
    {
        if (mu < 0)
        {
            throw new ArgumentException("Mu must be non-negative.", nameof(mu));
        }

        _mu = mu;
    }

    /// <summary>
    /// Aggregates client models using FedProx weighted averaging.
    /// </summary>
    /// <remarks>
    /// The aggregation step in FedProx is identical to FedAvg. The key difference is in
    /// the local training objective (which includes the proximal term), not in aggregation.
    ///
    /// <b>For Beginners:</b> At the server side, FedProx aggregates just like FedAvg.
    /// The magic happens during client-side training where the proximal term keeps
    /// client models from straying too far.
    ///
    /// Aggregation formula (same as FedAvg):
    /// w_global = Σ(n_k / n_total) × w_k
    ///
    /// The proximal term μ affects how w_k is computed during local training, but not
    /// how we aggregate the models here.
    ///
    /// For implementation in local training (not shown here):
    /// - Gradient = ∇Loss + μ(w - w_global)
    /// - This additional term pulls weights towards global model
    /// </remarks>
    /// <param name="clientModels">Dictionary mapping client IDs to their model parameters.</param>
    /// <param name="clientWeights">Dictionary mapping client IDs to their sample counts (weights).</param>
    /// <returns>The aggregated global model parameters.</returns>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the name of the aggregation strategy.
    /// </summary>
    /// <returns>A string indicating "FedProx" with the μ parameter value.</returns>
    public override string GetStrategyName()
    {
        return $"FedProx(μ={_mu})";
    }

    /// <summary>
    /// Gets the proximal term coefficient μ.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns the strength of the constraint that keeps client
    /// models from deviating too far from the global model.
    /// </remarks>
    /// <returns>The μ parameter value.</returns>
    public double GetMu()
    {
        return _mu;
    }
}
