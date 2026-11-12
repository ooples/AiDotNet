using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Coordinate-wise median aggregation for <see cref="IFullModel{T,TInput,TOutput}"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> For each model parameter, this strategy takes the middle value across clients.
/// This makes the aggregation resistant to outliers (e.g., a client sending extremely large values).
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class MedianFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    /// <inheritdoc />
    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        _ = clientWeights; // Unused: median is unweighted by design.

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        var sorted = SortCoordinatesAcrossClients(clientParameters, clientIds);
        var aggregated = (n % 2 == 1)
            ? OrderStatisticRow(sorted, n / 2)
            : Engine.Multiply(SumOrderStatisticRows(sorted, (n / 2) - 1, 2), NumOps.FromDouble(0.5));

        return InterfaceGuard.Parameterizable(reference).WithParameters(aggregated);
    }

    /// <inheritdoc />
    public override string GetStrategyName() => "Median";
}

