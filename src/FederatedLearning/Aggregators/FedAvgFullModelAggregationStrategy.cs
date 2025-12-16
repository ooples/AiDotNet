using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// FedAvg aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> using vector-based parameters.
/// </summary>
public sealed class FedAvgFullModelAggregationStrategy<T, TInput, TOutput> :
    AggregationStrategyBase<IFullModel<T, TInput, TOutput>, T>
{
    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));

        var first = clientModels.First().Value;
        var firstParams = first.GetParameters();

        var aggregated = new Vector<T>(firstParams.Length);
        for (int i = 0; i < aggregated.Length; i++)
        {
            aggregated[i] = NumOps.Zero;
        }

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;
            var model = kvp.Value;
            var parameters = model.GetParameters();
            if (parameters.Length != aggregated.Length)
            {
                throw new ArgumentException($"Parameter length mismatch for client {clientId}.", nameof(clientModels));
            }

            if (!clientWeights.TryGetValue(clientId, out var weight))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            var normalizedWeightT = NumOps.FromDouble(weight / totalWeight);
            for (int i = 0; i < aggregated.Length; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(parameters[i], normalizedWeightT));
            }
        }

        return first.WithParameters(aggregated);
    }

    public override string GetStrategyName() => "FedAvg";
}
