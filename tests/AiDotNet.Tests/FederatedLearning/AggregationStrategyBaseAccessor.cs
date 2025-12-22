namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Aggregators;

internal sealed class AggregationStrategyBaseAccessor : AggregationStrategyBase<int, double>
{
    public override int Aggregate(Dictionary<int, int> clientModels, Dictionary<int, double> clientWeights)
    {
        throw new NotSupportedException("Test accessor only.");
    }

    public override string GetStrategyName() => "Accessor";

    public static double CallGetTotalWeightOrThrow(
        Dictionary<int, double> clientWeights,
        IEnumerable<int> clientIds,
        string paramName)
    {
        return GetTotalWeightOrThrow(clientWeights, clientIds, paramName);
    }
}

