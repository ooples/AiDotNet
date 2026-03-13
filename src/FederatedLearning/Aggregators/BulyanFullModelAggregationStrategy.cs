using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Bulyan aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> (Multi-Krum selection + trimmed aggregation).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Bulyan is a stronger robust aggregation approach:
/// 1) Use Multi-Krum to pick a set of "reasonable" client updates.
/// 2) For each parameter, apply a trimmed aggregation over that set.
///
/// This can better tolerate malicious clients, at the cost of more computation.
/// </remarks>
public sealed class BulyanFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly int _byzantineClientCount;
    private readonly bool _useClientWeightsForAveraging;

    public BulyanFullModelAggregationStrategy(int byzantineClientCount = 1, bool useClientWeightsForAveraging = false)
    {
        if (byzantineClientCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byzantineClientCount), "Byzantine client count must be non-negative.");
        }

        _byzantineClientCount = byzantineClientCount;
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

        // A common requirement for Bulyan is n >= 4f + 3.
        if (n < (4 * f) + 3)
        {
            throw new InvalidOperationException($"Bulyan requires n >= 4f + 3. Got n={n}, f={f}.");
        }

        // Selection size for the Multi-Krum stage.
        int selectionSize = n - (2 * f);
        if (selectionSize <= 0)
        {
            throw new InvalidOperationException($"Invalid Bulyan selection size for n={n}, f={f}.");
        }

        var selectedClientIds = SelectMultiKrumCandidates(clientParameters, clientIds, f, selectionSize);

        // Trim within selected set by f on each side per coordinate (Bulyan-style).
        int m = selectedClientIds.Count;
        int trim = Math.Min(f, (m - 1) / 2);
        int kept = m - (2 * trim);
        if (kept <= 0)
        {
            throw new InvalidOperationException($"Bulyan trimming invalid for m={m}, f={f}.");
        }

        if (_useClientWeightsForAveraging)
        {
            // Weighting in Bulyan is not standard; provide as an optional mode by averaging the selected subset.
            var averagedParameters = WeightedAverageOrUnweightedAverage(selectedClientIds, clientParameters, clientWeights, useClientWeights: true);
            return reference.WithParameters(averagedParameters);
        }

        var aggregated = new Vector<T>(parameterCount);
        var buffer = new double[m];

        for (int i = 0; i < parameterCount; i++)
        {
            for (int j = 0; j < m; j++)
            {
                buffer[j] = NumOps.ToDouble(clientParameters[selectedClientIds[j]][i]);
            }

            Array.Sort(buffer);
            double sum = 0.0;
            for (int j = trim; j < m - trim; j++)
            {
                sum += buffer[j];
            }

            aggregated[i] = NumOps.FromDouble(sum / kept);
        }

        return reference.WithParameters(aggregated);
    }

    private List<int> SelectMultiKrumCandidates(
        Dictionary<int, Vector<T>> clientParameters,
        int[] clientIds,
        int byzantineClientCount,
        int selectionSize)
    {
        int n = clientIds.Length;
        int f = byzantineClientCount;
        int neighborsToSum = n - f - 2;
        if (neighborsToSum <= 0)
        {
            throw new InvalidOperationException($"Multi-Krum candidate selection requires at least f+3 clients. Got n={n}, f={f}.");
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

        return scores
            .OrderBy(s => s.Score)
            .ThenBy(s => s.ClientId)
            .Take(selectionSize)
            .Select(s => s.ClientId)
            .ToList();
    }

    public override string GetStrategyName() => "Bulyan";
}

