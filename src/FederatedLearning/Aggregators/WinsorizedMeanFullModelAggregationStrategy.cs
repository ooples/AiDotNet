using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Coordinate-wise winsorized mean aggregation for <see cref="IFullModel{T,TInput,TOutput}"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Winsorized mean is like trimmed mean, but instead of *dropping* extreme values,
/// it *clips* them to the nearest remaining value before averaging. This reduces the impact of outliers
/// while keeping the same number of values in the average.
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class WinsorizedMeanFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly double _winsorizeFraction;

    public WinsorizedMeanFullModelAggregationStrategy(double winsorizeFraction = 0.2)
    {
        if (winsorizeFraction < 0.0 || winsorizeFraction >= 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(winsorizeFraction), "Winsorize fraction must be in [0.0, 0.5).");
        }

        _winsorizeFraction = winsorizeFraction;
    }

    /// <inheritdoc />
    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        _ = clientWeights; // Unused: winsorized mean is unweighted by design.

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        int w = (int)Math.Floor(_winsorizeFraction * n);
        int lowerIndex = Math.Min(Math.Max(0, w), Math.Max(0, n - 1));
        int upperIndex = Math.Max(0, Math.Min(n - 1 - w, n - 1));
        if (lowerIndex > upperIndex)
        {
            throw new InvalidOperationException($"Winsorize fraction {_winsorizeFraction:0.###} is too large for {n} clients.");
        }

        // After sorting each coordinate, every order statistic below lowerIndex clamps to row lowerIndex and every
        // one above upperIndex clamps to row upperIndex, so the winsorized mean weights row lowerIndex by
        // (lowerIndex + 1) / n, row upperIndex by (n - upperIndex) / n and every row between by 1 / n.
        var sorted = SortCoordinatesAcrossClients(clientParameters, clientIds);
        var rowWeights = new double[n];
        for (int r = lowerIndex; r <= upperIndex; r++)
        {
            rowWeights[r] = 1.0 / n;
        }

        rowWeights[lowerIndex] += (double)lowerIndex / n;
        rowWeights[upperIndex] += (double)(n - 1 - upperIndex) / n;
        var aggregated = CombineOrderStatisticRows(sorted, rowWeights);

        return InterfaceGuard.Parameterizable(reference).WithParameters(aggregated);
    }

    /// <inheritdoc />
    public override string GetStrategyName() => "WinsorizedMean";
}

