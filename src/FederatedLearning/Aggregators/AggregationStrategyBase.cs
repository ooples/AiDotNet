namespace AiDotNet.FederatedLearning.Aggregators;

using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

/// <summary>
/// Base class for federated aggregation strategies.
/// </summary>
/// <typeparam name="TModel">Model/update representation.</typeparam>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class AggregationStrategyBase<TModel, T> : FederatedLearningComponentBase<T>, IAggregationStrategy<TModel>
{
    public abstract TModel Aggregate(Dictionary<int, TModel> clientModels, Dictionary<int, double> clientWeights);

    public abstract string GetStrategyName();

    protected static double GetTotalWeightOrThrow(
        Dictionary<int, double> clientWeights,
        IEnumerable<int> clientIds,
        string paramName)
    {
        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", paramName);
        }

        double total = 0.0;
        foreach (var clientId in clientIds)
        {
            if (!clientWeights.TryGetValue(clientId, out var w))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", paramName);
            }

            total += w;
        }

        if (total <= 0.0)
        {
            throw new ArgumentException("Total weight must be positive.", paramName);
        }

        return total;
    }
}

