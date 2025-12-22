using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// FedBN aggregation for <see cref="IFullModel{T,TInput,TOutput}"/> when the model is a <see cref="NeuralNetworkBase{T}"/>.
/// </summary>
/// <remarks>
/// This implementation keeps batch-normalization layer parameters local by copying them from the first client model,
/// while aggregating all other parameters using weighted averaging.
/// </remarks>
public sealed class FedBNFullModelAggregationStrategy<T, TInput, TOutput> :
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

        var first = clientModels.First().Value;
        if (first is not NeuralNetworkBase<T> firstNet)
        {
            // Non-neural-network models cannot be BN-split; fall back to FedAvg behavior.
            return new FedAvgFullModelAggregationStrategy<T, TInput, TOutput>().Aggregate(clientModels, clientWeights);
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));

        var firstParams = first.GetParameters();
        var bnRanges = GetBatchNormParameterRanges(firstNet);

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
                if (IsInRanges(i, bnRanges))
                {
                    continue;
                }

                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(parameters[i], normalizedWeightT));
            }
        }

        // For BN parameters, keep the first client's values (client-specific BN).
        foreach (var range in bnRanges)
        {
            for (int i = 0; i < range.Length; i++)
            {
                aggregated[range.Start + i] = firstParams[range.Start + i];
            }
        }

        return first.WithParameters(aggregated);
    }

    public override string GetStrategyName() => "FedBN";

    private static List<(int Start, int Length)> GetBatchNormParameterRanges(NeuralNetworkBase<T> network)
    {
        var ranges = new List<(int Start, int Length)>();

        int current = 0;
        foreach (var layer in network.Layers)
        {
            int count = layer.ParameterCount;
            if (count <= 0)
            {
                continue;
            }

            if (layer is BatchNormalizationLayer<T>)
            {
                ranges.Add((current, count));
            }

            current += count;
        }

        return ranges;
    }

    private static bool IsInRanges(int index, List<(int Start, int Length)> ranges)
    {
        for (int i = 0; i < ranges.Count; i++)
        {
            var r = ranges[i];
            if (index >= r.Start && index < r.Start + r.Length)
            {
                return true;
            }
        }

        return false;
    }
}
