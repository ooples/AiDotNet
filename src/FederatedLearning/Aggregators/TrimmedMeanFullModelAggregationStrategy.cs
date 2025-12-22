using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Coordinate-wise trimmed mean aggregation for <see cref="IFullModel{T,TInput,TOutput}"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This strategy sorts each parameter across clients, drops the extreme values
/// on both ends, then averages the remaining values. This reduces the impact of outliers.
/// </remarks>
public sealed class TrimmedMeanFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly double _trimFraction;

    public TrimmedMeanFullModelAggregationStrategy(double trimFraction = 0.2)
    {
        if (trimFraction < 0.0 || trimFraction >= 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(trimFraction), "Trim fraction must be in [0.0, 0.5).");
        }

        _trimFraction = trimFraction;
    }

    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        _ = clientWeights; // Unused: trimmed mean is unweighted by design.

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        int trim = (int)Math.Floor(_trimFraction * n);
        int kept = n - (2 * trim);
        if (kept <= 0)
        {
            throw new InvalidOperationException($"Trim fraction {_trimFraction:0.###} is too large for {n} clients.");
        }

        var aggregated = new Vector<T>(parameterCount);
        var buffer = new double[n];

        for (int i = 0; i < parameterCount; i++)
        {
            for (int j = 0; j < n; j++)
            {
                buffer[j] = NumOps.ToDouble(clientParameters[clientIds[j]][i]);
            }

            Array.Sort(buffer);
            double sum = 0.0;
            for (int j = trim; j < n - trim; j++)
            {
                sum += buffer[j];
            }

            aggregated[i] = NumOps.FromDouble(sum / kept);
        }

        return reference.WithParameters(aggregated);
    }

    public override string GetStrategyName() => "TrimmedMean";
}

