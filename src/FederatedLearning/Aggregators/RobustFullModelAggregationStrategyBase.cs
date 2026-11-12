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
        return Engine.TensorSort(stacked, 0, false).Values;
    }

    /// <summary>Returns row <paramref name="row"/> of an [n, P] tensor as a length-P vector.</summary>
    private protected static Vector<T> OrderStatisticRow(Tensor<T> sorted, int row)
    {
        int p = sorted.Shape[1];
        return Engine.TensorSlice(sorted, new[] { row, 0 }, new[] { 1, p }).ToVector();
    }

    /// <summary>Sums rows [<paramref name="start"/>, <paramref name="start"/> + <paramref name="count"/>) of an [n, P] tensor.</summary>
    private protected static Vector<T> SumOrderStatisticRows(Tensor<T> sorted, int start, int count)
    {
        int p = sorted.Shape[1];
        var block = Engine.TensorSlice(sorted, new[] { start, 0 }, new[] { count, p });
        return Engine.ReduceSum(block, new[] { 0 }, false).ToVector();
    }

    protected double ComputeSquaredL2Distance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector length mismatch when computing distance.");
        }

        var diff = Engine.Subtract(a, b);
        return NumOps.ToDouble(Engine.DotProduct(diff, diff));
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

        int parameterCount = clientParameters[selectedClientIds[0]].Length;
        var aggregated = new Vector<T>(parameterCount);

        if (!useClientWeights)
        {
            var denom = NumOps.FromDouble(selectedClientIds.Count);
            foreach (var clientId in selectedClientIds)
            {
                aggregated = Engine.Add(aggregated, clientParameters[clientId]);
            }

            return Engine.Divide(aggregated, denom);
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, selectedClientIds, nameof(clientWeights));
        foreach (var clientId in selectedClientIds)
        {
            if (!clientWeights.TryGetValue(clientId, out var weight))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            var normalizedWeightT = NumOps.FromDouble(weight / totalWeight);
            aggregated = Engine.Add(aggregated, Engine.Multiply(clientParameters[clientId], normalizedWeightT));
        }

        return aggregated;
    }
}
