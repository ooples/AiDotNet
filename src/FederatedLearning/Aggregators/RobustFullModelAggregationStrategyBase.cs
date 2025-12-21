using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Base class for robust aggregation strategies operating on <see cref="IFullModel{T,TInput,TOutput}"/> parameters.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <typeparam name="TInput">Model input type.</typeparam>
/// <typeparam name="TOutput">Model output type.</typeparam>
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
        var parameters = reference.GetParameters();
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
            var p = kvp.Value.GetParameters();
            if (p.Length != expectedParameterCount)
            {
                throw new ArgumentException($"Parameter length mismatch for client {kvp.Key}.", nameof(clientModels));
            }

            parameters[kvp.Key] = p;
        }

        return parameters;
    }

    protected double ComputeSquaredL2Distance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector length mismatch when computing distance.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.ToDouble(sum);
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
        for (int i = 0; i < aggregated.Length; i++)
        {
            aggregated[i] = NumOps.Zero;
        }

        if (!useClientWeights)
        {
            var denom = NumOps.FromDouble(selectedClientIds.Count);
            for (int i = 0; i < parameterCount; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < selectedClientIds.Count; j++)
                {
                    sum += NumOps.ToDouble(clientParameters[selectedClientIds[j]][i]);
                }

                aggregated[i] = NumOps.Divide(NumOps.FromDouble(sum), denom);
            }

            return aggregated;
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, selectedClientIds, nameof(clientWeights));
        for (int i = 0; i < parameterCount; i++)
        {
            T sum = NumOps.Zero;
            foreach (var clientId in selectedClientIds)
            {
                if (!clientWeights.TryGetValue(clientId, out var weight))
                {
                    throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
                }

                var normalizedWeightT = NumOps.FromDouble(weight / totalWeight);
                sum = NumOps.Add(sum, NumOps.Multiply(clientParameters[clientId][i], normalizedWeightT));
            }

            aggregated[i] = sum;
        }

        return aggregated;
    }
}
