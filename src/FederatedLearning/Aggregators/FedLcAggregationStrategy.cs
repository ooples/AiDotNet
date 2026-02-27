namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedLC (Logit Calibration) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When different clients have different class distributions
/// (e.g., Hospital A sees mostly flu, Hospital B sees mostly cold), local models develop
/// biased predictions. FedLC fixes this by adjusting each client's logits based on its
/// local class frequency before aggregation.</para>
///
/// <para>During local training, logits are calibrated:</para>
/// <code>z_calibrated[c] = z[c] - τ * log(p_local[c])</code>
/// <para>where p_local[c] is the local class frequency for class c.</para>
///
/// <para>Reference: Zhang, J., et al. (2022). "Federated Learning with Label Distribution Skew
/// via Logits Calibration." ICML 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedLcAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _calibrationTemperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedLcAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="calibrationTemperature">Temperature for logit calibration. Default: 1.0 per paper.</param>
    public FedLcAggregationStrategy(double calibrationTemperature = 1.0)
    {
        if (calibrationTemperature <= 0)
        {
            throw new ArgumentException("Calibration temperature must be positive.", nameof(calibrationTemperature));
        }

        _calibrationTemperature = calibrationTemperature;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the logit calibration temperature.
    /// </summary>
    public double CalibrationTemperature => _calibrationTemperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedLC(τ={_calibrationTemperature})";
}
