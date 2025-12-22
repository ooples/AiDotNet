using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Krum aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> (Byzantine-robust selection by distance).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Krum picks the single client update that is most consistent with the others.
/// It does this by computing distances between client updates and selecting the one with the smallest
/// sum of distances to its closest neighbors.
/// </remarks>
public sealed class KrumFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly int _byzantineClientCount;

    public KrumFullModelAggregationStrategy(int byzantineClientCount = 1)
    {
        if (byzantineClientCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byzantineClientCount), "Byzantine client count must be non-negative.");
        }

        _byzantineClientCount = byzantineClientCount;
    }

    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        _ = clientWeights; // Unused: selection does not use weights.

        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();
        int n = clientIds.Length;

        // Krum requires n >= f + 3 so that (n - f - 2) >= 1.
        int f = _byzantineClientCount;
        int neighborsToSum = n - f - 2;
        if (neighborsToSum <= 0)
        {
            throw new InvalidOperationException($"Krum requires at least f+3 clients. Got n={n}, f={f}.");
        }

        int bestClientId = clientIds[0];
        double bestScore = double.PositiveInfinity;

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

            if (score < bestScore)
            {
                bestScore = score;
                bestClientId = clientI;
            }
        }

        var selectedParameters = clientParameters[bestClientId];
        return reference.WithParameters(selectedParameters);
    }

    public override string GetStrategyName() => "Krum";
}

