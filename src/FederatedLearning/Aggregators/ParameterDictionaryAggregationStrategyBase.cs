
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

                AddScaledInto(aggregatedParams, clientParams, normalizedWeightT);
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
            AddScaledInto(destination, clientParams, normalizedWeightT);
        }
    }

    protected static T[] CreateZeroInitializedLayer(int length)
    {
        var layer = new T[length];
        Vector<T>.Wrap(layer).Fill(NumOps.Zero);
        return layer;
    }

    /// <summary>Computes <c>destination += source * weight</c> with engine ops and writes it back into <paramref name="destination"/>.</summary>
    private protected static void AddScaledInto(T[] destination, T[] source, T weight)
    {
        var sum = Engine.Add(Vector<T>.Wrap(destination), Engine.Multiply(Vector<T>.Wrap(source), weight));
        sum.AsSpan().CopyTo(destination);
    }

    /// <summary>
    /// Concatenates a client's layers (in <paramref name="layerNames"/> order) into one parameter vector.
    /// </summary>
    private protected static Vector<T> FlattenLayers(Dictionary<string, T[]> model, string[] layerNames, int totalParams)
    {
        var flat = new T[totalParams];
        int offset = 0;
        foreach (var layerName in layerNames)
        {
            var layer = model[layerName];
            Array.Copy(layer, 0, flat, offset, layer.Length);
            offset += layer.Length;
        }

        return Vector<T>.Wrap(flat);
    }
}
