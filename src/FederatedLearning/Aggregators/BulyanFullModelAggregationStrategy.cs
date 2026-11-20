using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Bulyan aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> (Multi-Krum selection + closest-to-median averaging).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Bulyan is a stronger robust aggregation approach:
/// 1) Use Multi-Krum to pick a set of "reasonable" client updates.
/// 2) For each parameter, average the n - 4f selected values closest to that parameter's median.
///
/// This can better tolerate malicious clients, at the cost of more computation.
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

    /// <inheritdoc />
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

        // Bulyan (El Mhamdi et al., 2018), Alg. 1 step 2: for every coordinate, average the beta = theta - 2f
        // selected values closest to that coordinate's median.
        int m = selectedClientIds.Count;
        int beta = m - (2 * f);
        if (beta <= 0)
        {
            throw new InvalidOperationException($"Bulyan trimming invalid for m={m}, f={f}.");
        }

        if (_useClientWeightsForAveraging)
        {
            // Weighting in Bulyan is not standard; provide as an optional mode by averaging the selected subset.
            var averagedParameters = WeightedAverageOrUnweightedAverage(selectedClientIds, clientParameters, clientWeights, useClientWeights: true);
            return InterfaceGuard.Parameterizable(reference).WithParameters(averagedParameters);
        }

        var sorted = SortCoordinatesAcrossClients(clientParameters, selectedClientIds.ToArray());

        // A narrow T (Half, BFloat16) is evaluated in float: |x - median| and the beta-term sum can pass 65504.
        var aggregated = AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.IsNarrowFloat<T>()
            ? ClosestToMedianMean(sorted.Cast<float>(), beta).Cast<T>().ToVector()
            : ClosestToMedianMean(sorted, beta).ToVector();

        return InterfaceGuard.Parameterizable(reference).WithParameters(aggregated);
    }

    /// <summary>
    /// Per coordinate (column) of an [m, P] tensor of sorted values, the mean of the <paramref name="beta"/>
    /// values closest to the column's median (the mean of the two middle values for an even m).
    /// </summary>
    private static Tensor<TV> ClosestToMedianMean<TV>(Tensor<TV> sorted, int beta)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();
        int m = sorted.Shape[0];
        int p = sorted.Shape[1];
        var upper = Engine.TensorSlice(sorted, new[] { m / 2, 0 }, new[] { 1, p });
        var median = m % 2 == 1
            ? upper
            : Engine.TensorAdd(
                Engine.TensorMultiplyScalar(Engine.TensorSlice(sorted, new[] { (m / 2) - 1, 0 }, new[] { 1, p }), ops.FromDouble(0.5)),
                Engine.TensorMultiplyScalar(upper, ops.FromDouble(0.5)));
        var distanceToMedian = Engine.TensorAbs(Engine.TensorSubtract(sorted, Engine.TensorBroadcastTo(median, new[] { m, p })));
        Engine.TensorTopK(Engine.TensorNegate(distanceToMedian), beta, 0, out var closestRows);
        var closest = Engine.TensorTakeAlongDim(sorted, closestRows, 0);
        return Engine.ReduceSum(Engine.TensorMultiplyScalar(closest, ops.FromDouble(1.0 / beta)), new[] { 0 }, false);
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

    /// <inheritdoc />
    public override string GetStrategyName() => "Bulyan";
}

