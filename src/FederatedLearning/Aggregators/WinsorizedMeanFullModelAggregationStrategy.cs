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

        var aggregated = new Vector<T>(parameterCount);
        var buffer = new double[n];

        for (int i = 0; i < parameterCount; i++)
        {
            for (int j = 0; j < n; j++)
            {
                buffer[j] = NumOps.ToDouble(clientParameters[clientIds[j]][i]);
            }

            Array.Sort(buffer);

            double lower = buffer[lowerIndex];
            double upper = buffer[upperIndex];
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                double v = buffer[j];
                if (v < lower)
                {
                    v = lower;
                }
                else if (v > upper)
                {
                    v = upper;
                }

                sum += v;
            }

            aggregated[i] = NumOps.FromDouble(sum / n);
        }

        return reference.WithParameters(aggregated);
    }

    public override string GetStrategyName() => "WinsorizedMean";
}

