using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Multi-Krum aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> (select m central updates, then average).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Multi-Krum is like Krum, but instead of picking only one client update,
/// it picks a small group of the most "central" updates and averages them.
/// </remarks>
public sealed class MultiKrumFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly int _byzantineClientCount;
    private readonly int _selectionCount;
    private readonly bool _useClientWeightsForAveraging;

    public MultiKrumFullModelAggregationStrategy(
        int byzantineClientCount = 1,
        int selectionCount = 0,
        bool useClientWeightsForAveraging = false)
    {
        if (byzantineClientCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byzantineClientCount), "Byzantine client count must be non-negative.");
        }

        _byzantineClientCount = byzantineClientCount;
        _selectionCount = selectionCount;
        _useClientWeightsForAveraging = useClientWeightsForAveraging;
    }

    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        int f = _byzantineClientCount;
        int neighborsToSum = n - f - 2;
        if (neighborsToSum <= 0)
        {
            throw new InvalidOperationException($"Multi-Krum requires at least f+3 clients. Got n={n}, f={f}.");
        }

        int m = _selectionCount > 0 ? _selectionCount : Math.Max(1, n - (2 * f) - 2);
        if (m <= 0)
        {
            throw new InvalidOperationException($"Invalid Multi-Krum selection count m={m} for n={n}, f={f}.");
        }

        if (m > n)
        {
            m = n;
        }

        var scores = new List<(int ClientId, double Score)>(n);

        for (int i = 0; i < n; i++)
        {
            int clientI = clientIds[i];
            var distances = new double[n - 1];
            int idx = 0;
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }

                int clientJ = clientIds[j];
                distances[idx++] = ComputeSquaredL2Distance(clientParameters[clientI], clientParameters[clientJ]);
            }

            Array.Sort(distances);
            double score = 0.0;
            for (int k = 0; k < neighborsToSum; k++)
            {
                score += distances[k];
            }

            scores.Add((clientI, score));
        }

        var selectedClientIds = scores
            .OrderBy(s => s.Score)
            .ThenBy(s => s.ClientId)
            .Take(m)
            .Select(s => s.ClientId)
            .ToList();

        var aggregatedParameters = WeightedAverageOrUnweightedAverage(
            selectedClientIds,
            clientParameters,
            clientWeights,
            useClientWeights: _useClientWeightsForAveraging);

        return reference.WithParameters(aggregatedParameters);
    }

    public override string GetStrategyName() => "MultiKrum";
}

