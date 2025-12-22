using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Coordinate-wise median aggregation for <see cref="IFullModel{T,TInput,TOutput}"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> For each model parameter, this strategy takes the middle value across clients.
/// This makes the aggregation resistant to outliers (e.g., a client sending extremely large values).
/// </remarks>
public sealed class MedianFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        _ = clientWeights; // Unused: median is unweighted by design.

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        var aggregated = new Vector<T>(parameterCount);
        var buffer = new double[n];

        for (int i = 0; i < parameterCount; i++)
        {
            for (int j = 0; j < n; j++)
            {
                buffer[j] = NumOps.ToDouble(clientParameters[clientIds[j]][i]);
            }

            Array.Sort(buffer);
            double median = (n % 2 == 1)
                ? buffer[n / 2]
                : 0.5 * (buffer[(n / 2) - 1] + buffer[n / 2]);

            aggregated[i] = NumOps.FromDouble(median);
        }

        return reference.WithParameters(aggregated);
    }

    public override string GetStrategyName() => "Median";
}

