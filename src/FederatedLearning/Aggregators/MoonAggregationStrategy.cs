namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements MOON (Model-COntrastive Learning) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MOON corrects "local drift" by adding a contrastive loss
/// during client training. The contrastive loss pulls the local model's representation closer
/// to the global model and pushes it away from the previous local model, reducing divergence
/// caused by non-IID data.</para>
///
/// <para>During aggregation, MOON uses standard weighted averaging (same as FedAvg). The key
/// innovation is in the local training objective, which includes:</para>
/// <code>L_total = L_task + μ * L_contrastive(z_local, z_global, z_prev_local)</code>
///
/// <para>Reference: Li, Q., He, B., and Song, D. (2021). "Model-Contrastive Federated Learning."
/// CVPR 2021.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class MoonAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _contrastiveWeight;
    private readonly double _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="MoonAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="contrastiveWeight">Weight of the contrastive loss term (mu). Default: 1.0 per paper.</param>
    /// <param name="temperature">Temperature for contrastive similarity. Default: 0.5 per paper.</param>
    public MoonAggregationStrategy(double contrastiveWeight = 1.0, double temperature = 0.5)
    {
        if (contrastiveWeight < 0)
        {
            throw new ArgumentException("Contrastive weight must be non-negative.", nameof(contrastiveWeight));
        }

        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive.", nameof(temperature));
        }

        _contrastiveWeight = contrastiveWeight;
        _temperature = temperature;
    }

    /// <summary>
    /// Aggregates client models using weighted averaging.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The server-side aggregation in MOON is identical to FedAvg.
    /// The contrastive loss is applied during local training on each client, not during aggregation.</para>
    /// </remarks>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the contrastive loss weight (mu).
    /// </summary>
    public double ContrastiveWeight => _contrastiveWeight;

    /// <summary>
    /// Gets the contrastive temperature parameter.
    /// </summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"MOON(μ={_contrastiveWeight},τ={_temperature})";
}
