namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedNTD (Not-True Distillation) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedNTD addresses the problem of local models "forgetting" the
/// global model's knowledge during local training. It adds a distillation loss that only
/// penalizes changes to non-true class logits (the classes that are NOT the correct answer),
/// preserving local knowledge about the true class while keeping the rest aligned with the
/// global model.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_CE + β * KL(softmax(z_global / τ) || softmax(z_local / τ)) [non-true classes only]</code>
///
/// <para>Reference: Lee, G., Shin, M., and Hwang, S. J. (2022). "Preservation of the Global Knowledge
/// by Not-True Distillation in Federated Learning." NeurIPS 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedNtdAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _distillationWeight;
    private readonly double _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedNtdAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="distillationWeight">Weight of the not-true distillation loss (beta). Default: 1.0 per paper.</param>
    /// <param name="temperature">Softmax temperature for distillation. Default: 3.0 per paper.</param>
    public FedNtdAggregationStrategy(double distillationWeight = 1.0, double temperature = 3.0)
    {
        if (distillationWeight < 0)
        {
            throw new ArgumentException("Distillation weight must be non-negative.", nameof(distillationWeight));
        }

        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive.", nameof(temperature));
        }

        _distillationWeight = distillationWeight;
        _temperature = temperature;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the not-true distillation weight (beta).
    /// </summary>
    public double DistillationWeight => _distillationWeight;

    /// <summary>
    /// Gets the softmax temperature for distillation.
    /// </summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedNTD(β={_distillationWeight},τ={_temperature})";
}
