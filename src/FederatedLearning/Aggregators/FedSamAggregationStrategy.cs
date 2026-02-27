namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Specifies the FedSAM variant to use.
/// </summary>
public enum FedSamVariant
{
    /// <summary>Base FedSAM — standard sharpness-aware minimization per client.</summary>
    Base,
    /// <summary>FedSMOO — simultaneous global and local flatness optimization.</summary>
    FedSMOO,
    /// <summary>FedSpeed — efficient SAM with gradient perturbation approximation.</summary>
    FedSpeed,
    /// <summary>FedLESAM — locally estimated SAM to reduce overhead.</summary>
    FedLESAM,
    /// <summary>FedSCAM — stochastic controlled averaging for SAM in FL.</summary>
    FedSCAM
}

/// <summary>
/// Implements FedSAM (Sharpness-Aware Minimization for Federated Learning) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Regular optimization finds a low point (minimum) in the loss
/// landscape, but this point might be "sharp" — a small change in parameters causes a large
/// change in loss. FedSAM instead seeks "flat" minima that are more robust, which is especially
/// important in FL where each client's data creates a different loss landscape.</para>
///
/// <para>Local training uses a two-step process per batch:</para>
/// <code>
/// 1. Compute gradient, take a step in gradient direction (perturbation)
/// 2. Compute gradient at the perturbed point, use that for the actual update
/// </code>
///
/// <para>Variants reduce the overhead of this two-gradient computation or optimize for both
/// global and local flatness simultaneously.</para>
///
/// <para>Reference: Caldarola, D., et al. (2022). "Improving Generalization in Federated Learning
/// by Seeking Flat Minima."</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedSamAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _perturbationRadius;
    private readonly FedSamVariant _variant;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedSamAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="perturbationRadius">Radius for the SAM perturbation step (rho). Default: 0.05 per paper.</param>
    /// <param name="variant">The FedSAM variant to use. Default: Base.</param>
    public FedSamAggregationStrategy(double perturbationRadius = 0.05, FedSamVariant variant = FedSamVariant.Base)
    {
        if (perturbationRadius <= 0)
        {
            throw new ArgumentException("Perturbation radius must be positive.", nameof(perturbationRadius));
        }

        _perturbationRadius = perturbationRadius;
        _variant = variant;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the SAM perturbation radius (rho).
    /// </summary>
    public double PerturbationRadius => _perturbationRadius;

    /// <summary>
    /// Gets the FedSAM variant being used.
    /// </summary>
    public FedSamVariant Variant => _variant;

    /// <inheritdoc/>
    public override string GetStrategyName() => _variant == FedSamVariant.Base
        ? $"FedSAM(ρ={_perturbationRadius})"
        : $"FedSAM-{_variant}(ρ={_perturbationRadius})";
}
