namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedDecorr (Decorrelation) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In federated learning with non-IID data, local models tend to learn
/// redundant (correlated) features — a phenomenon called "dimensional collapse." FedDecorr adds a
/// decorrelation regularizer that encourages each client's feature representations to be diverse
/// and complementary, improving the quality of the aggregated global model.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_task + λ * ||C - I||_F²</code>
/// <para>where C is the correlation matrix of feature representations and I is the identity matrix.</para>
///
/// <para>Reference: Shi, Y., et al. (2023). "Towards Understanding and Mitigating Dimensional Collapse
/// in Heterogeneous Federated Learning." ICML 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedDecorrAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _decorrelationWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedDecorrAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="decorrelationWeight">Weight of the decorrelation loss (lambda). Default: 0.1 per paper.</param>
    public FedDecorrAggregationStrategy(double decorrelationWeight = 0.1)
    {
        if (decorrelationWeight < 0)
        {
            throw new ArgumentException("Decorrelation weight must be non-negative.", nameof(decorrelationWeight));
        }

        _decorrelationWeight = decorrelationWeight;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the decorrelation regularization weight (lambda).
    /// </summary>
    public double DecorrelationWeight => _decorrelationWeight;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedDecorr(λ={_decorrelationWeight})";
}
