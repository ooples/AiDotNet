namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Base class for aggregation strategies operating on parameter dictionaries.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class ParameterDictionaryAggregationStrategyBase<T> :
    AggregationStrategyBase<Dictionary<string, T[]>, T>
{
    protected Dictionary<string, T[]> AggregateWeightedAverage(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
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

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        var aggregatedModel = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            aggregatedModel[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;
            var clientModel = kvp.Value;

            if (!clientWeights.TryGetValue(clientId, out var clientWeight))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            var normalizedWeightT = NumOps.FromDouble(clientWeight / totalWeight);

            foreach (var layerName in layerNames)
            {
                if (!clientModel.TryGetValue(layerName, out var clientParams))
                {
                    throw new ArgumentException($"Client {clientId} is missing layer '{layerName}'.", nameof(clientModels));
                }

                var aggregatedParams = aggregatedModel[layerName];
                if (clientParams.Length != aggregatedParams.Length)
                {
                    throw new ArgumentException(
                        $"Layer '{layerName}' length mismatch for client {clientId}. Expected {aggregatedParams.Length}, got {clientParams.Length}.",
                        nameof(clientModels));
                }

                for (int i = 0; i < clientParams.Length; i++)
                {
                    aggregatedParams[i] = NumOps.Add(
                        aggregatedParams[i],
                        NumOps.Multiply(clientParams[i], normalizedWeightT));
                }
            }
        }

        return aggregatedModel;
    }

    protected void AggregateLayerWeightedAverageInto(
        string layerName,
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights,
        double totalWeight,
        T[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;
            var clientModel = kvp.Value;

            if (!clientWeights.TryGetValue(clientId, out var clientWeight))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            if (!clientModel.TryGetValue(layerName, out var clientParams))
            {
                throw new ArgumentException($"Client {clientId} is missing layer '{layerName}'.", nameof(clientModels));
            }

            if (clientParams.Length != destination.Length)
            {
                throw new ArgumentException(
                    $"Layer '{layerName}' length mismatch for client {clientId}. Expected {destination.Length}, got {clientParams.Length}.",
                    nameof(clientModels));
            }

            var normalizedWeightT = NumOps.FromDouble(clientWeight / totalWeight);
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = NumOps.Add(
                    destination[i],
                    NumOps.Multiply(clientParams[i], normalizedWeightT));
            }
        }
    }

    protected static T[] CreateZeroInitializedLayer(int length)
    {
        var layer = new T[length];
        for (int i = 0; i < layer.Length; i++)
        {
            layer[i] = NumOps.Zero;
        }

        return layer;
    }
}
