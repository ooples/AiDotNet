namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedAlign (Feature Alignment) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different clients with different data can learn different "languages"
/// for representing the same concepts. FedAlign adds a regularizer during local training that
/// forces each client's feature space to align with shared anchor representations, so all clients
/// speak the same "language" when their models are combined.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_task + α * D(f_local(anchors), f_global(anchors))</code>
/// <para>where anchors are shared reference inputs and D measures representation distance.</para>
///
/// <para>Reference: Mendieta, M., et al. (2022). "Local Learning Matters: Rethinking Data Heterogeneity
/// in Federated Learning." CVPR 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedAlignAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _alignmentWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedAlignAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="alignmentWeight">Weight of the alignment loss (alpha). Default: 1.0 per paper.</param>
    public FedAlignAggregationStrategy(double alignmentWeight = 1.0)
    {
        if (alignmentWeight < 0)
        {
            throw new ArgumentException("Alignment weight must be non-negative.", nameof(alignmentWeight));
        }

        _alignmentWeight = alignmentWeight;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the feature alignment weight (alpha).
    /// </summary>
    public double AlignmentWeight => _alignmentWeight;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedAlign(α={_alignmentWeight})";
}
