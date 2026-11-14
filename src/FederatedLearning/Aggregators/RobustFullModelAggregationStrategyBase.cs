using AiDotNet.Helpers;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Base class for robust aggregation strategies operating on <see cref="IFullModel{T,TInput,TOutput}"/> parameters.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <typeparam name="TInput">Model input type.</typeparam>
/// <typeparam name="TOutput">Model output type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class RobustFullModelAggregationStrategyBase<T, TInput, TOutput> :
    AggregationStrategyBase<IFullModel<T, TInput, TOutput>, T>
{
    protected static (IFullModel<T, TInput, TOutput> ReferenceModel, int ParameterCount) GetReferenceModelOrThrow(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        var reference = clientModels.First().Value;
        var parameters = InterfaceGuard.Parameterizable(reference).GetParameters();
        if (parameters == null)
        {
            throw new ArgumentException("Client model parameters cannot be null.", nameof(clientModels));
        }

        return (reference, parameters.Length);
    }

    protected static Dictionary<int, Vector<T>> GetClientParametersOrThrow(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        int expectedParameterCount)
    {
        var parameters = new Dictionary<int, Vector<T>>(clientModels.Count);
        foreach (var kvp in clientModels)
        {
            var p = InterfaceGuard.Parameterizable(kvp.Value).GetParameters();
            if (p.Length != expectedParameterCount)
            {
                throw new ArgumentException($"Parameter length mismatch for client {kvp.Key}.", nameof(clientModels));
            }

            parameters[kvp.Key] = p;
        }

        return parameters;
    }

    /// <summary>
    /// Stacks the clients' parameter vectors into an [n, P] tensor (rows in <paramref name="clientIds"/>
    /// order) and sorts every coordinate column ascending, so row r holds each coordinate's r-th order statistic.
    /// </summary>
    private protected static Tensor<T> SortCoordinatesAcrossClients(
        Dictionary<int, Vector<T>> clientParameters,
        int[] clientIds)
    {
        var rows = clientIds.Select(id => clientParameters[id].AsTensor()).ToArray();
        var stacked = Engine.TensorStack(rows, 0);

        // A Byzantine client may send NaN. Array.Sort (the per-coordinate sort this replaced) orders NaN below
        // every number, so it falls in the trimmed lower tail; the engine sort gives NaN no defined position.
        // NaN is therefore sorted as -Infinity. (A NaN anywhere makes the total NaN, so the common all-finite
        // case pays one reduction and no copy.)
        if (NumOps.IsNaN(Engine.TensorSum(stacked)))
        {
            stacked = Engine.TensorNanToNum(
                stacked, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        }

        return Engine.TensorSort(stacked, 0, false).Values;
    }

    /// <summary>Returns row <paramref name="row"/> of an [n, P] tensor as a length-P vector.</summary>
    private protected static Vector<T> OrderStatisticRow(Tensor<T> sorted, int row)
    {
        int p = sorted.Shape[1];
        return Engine.TensorSlice(sorted, new[] { row, 0 }, new[] { 1, p }).ToVector();
    }

    /// <summary>
    /// sum_r rowWeights[r] * row r of an [n, P] tensor of sorted order statistics, as a length-P vector.
    /// Means and clamped (winsorized) means are formed with the 1/count factor folded into the row weights, and
    /// a narrow T (Half, BFloat16) is combined in float, so a client average cannot overflow or saturate.
    /// </summary>
    /// <remarks>
    /// Rows with weight 0 are the trimmed order statistics. They are removed before the product rather than
    /// multiplied by 0: a Byzantine client's +/-Infinity (or NaN) sorts into exactly those rows, and
    /// 0 * Infinity is NaN, which would poison every coordinate the robust aggregator exists to protect.
    /// </remarks>
    private protected static Vector<T> CombineOrderStatisticRows(Tensor<T> sorted, double[] rowWeights)
    {
        if (rowWeights.Length != sorted.Shape[0])
        {
            throw new ArgumentException(
                $"Expected {sorted.Shape[0]} row weights, got {rowWeights.Length}.", nameof(rowWeights));
        }

        int[] kept = Enumerable.Range(0, rowWeights.Length).Where(r => rowWeights[r] != 0.0).ToArray();
        if (kept.Length == rowWeights.Length)
        {
            return FederatedReductions.WeightedRowSum(sorted, rowWeights);
        }

        if (kept.Length == 0)
        {
            return new Vector<T>(sorted.Shape[1]);
        }

        var keptRows = Engine.TensorIndexSelect(sorted, Vector<int>.Wrap(kept).AsTensor(), 0);
        return FederatedReductions.WeightedRowSum(keptRows, kept.Select(r => rowWeights[r]).ToArray());
    }

    /// <summary>Row weights that average rows [<paramref name="start"/>, <paramref name="start"/> + <paramref name="count"/>) of <paramref name="rows"/> rows.</summary>
    private protected static double[] BlockMeanWeights(int rows, int start, int count)
    {
        var weights = new double[rows];
        for (int r = start; r < start + count; r++)
        {
            weights[r] = 1.0 / count;
        }

        return weights;
    }

    protected double ComputeSquaredL2Distance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector length mismatch when computing distance.");
        }

        return FederatedReductions.SquaredDistance(a, b);
    }

    protected Vector<T> WeightedAverageOrUnweightedAverage(
        IReadOnlyList<int> selectedClientIds,
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        bool useClientWeights)
    {
        if (selectedClientIds == null || selectedClientIds.Count == 0)
        {
            throw new ArgumentException("Selected client IDs cannot be null or empty.", nameof(selectedClientIds));
        }

        var missing = selectedClientIds.Where(id => !clientParameters.ContainsKey(id)).ToList();
        if (missing.Count > 0)
        {
            throw new ArgumentException($"Missing parameters for selected clients: {string.Join(", ", missing)}.", nameof(selectedClientIds));
        }

        var vectors = selectedClientIds.Select(id => clientParameters[id]).ToArray();
        if (!useClientWeights)
        {
            return FederatedReductions.Mean(vectors);
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, selectedClientIds, nameof(clientWeights));
        var normalizedWeights = new double[selectedClientIds.Count];
        for (int c = 0; c < normalizedWeights.Length; c++)
        {
            if (!clientWeights.TryGetValue(selectedClientIds[c], out var weight))
            {
                throw new ArgumentException($"Missing weight for client {selectedClientIds[c]}.", nameof(clientWeights));
            }

            normalizedWeights[c] = weight / totalWeight;
        }

        return FederatedReductions.WeightedSum(vectors, normalizedWeights);
    }
}
