using AiDotNet.FederatedLearning.Aggregators;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FedProxAggregationStrategyTests
{
    [Fact]
    public void Aggregate_ReturnsWeightedAverage()
    {
        var aggregator = new FedProxAggregationStrategy<double>(mu: 0.5);

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = new Dictionary<string, double[]>
            {
                ["w"] = new[] { 1.0, 2.0 }
            },
            [2] = new Dictionary<string, double[]>
            {
                ["w"] = new[] { 3.0, 4.0 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [1] = 1.0,
            [2] = 3.0
        };

        var aggregated = aggregator.Aggregate(clientModels, clientWeights);

        Assert.Single(aggregated);
        Assert.Equal(2.5, aggregated["w"][0], precision: 10);
        Assert.Equal(3.5, aggregated["w"][1], precision: 10);

        Assert.Contains("FedProx", aggregator.GetStrategyName());
        Assert.Equal(0.5, aggregator.GetMu(), precision: 10);
    }
}

