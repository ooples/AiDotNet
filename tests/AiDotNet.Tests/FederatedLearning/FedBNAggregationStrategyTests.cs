using AiDotNet.FederatedLearning.Aggregators;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FedBNAggregationStrategyTests
{
    [Fact]
    public void Aggregate_KeepsBatchNormLayersLocal()
    {
        var aggregator = new FedBNAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = new Dictionary<string, double[]>
            {
                ["conv1"] = new[] { 1.0, 1.0 },
                ["bn1_gamma"] = new[] { 10.0, 10.0 }
            },
            [2] = new Dictionary<string, double[]>
            {
                ["conv1"] = new[] { 3.0, 3.0 },
                ["bn1_gamma"] = new[] { 20.0, 20.0 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [1] = 1.0,
            [2] = 1.0
        };

        var aggregated = aggregator.Aggregate(clientModels, clientWeights);

        Assert.Equal(2, aggregated.Count);
        Assert.Equal(2.0, aggregated["conv1"][0], precision: 10);
        Assert.Equal(2.0, aggregated["conv1"][1], precision: 10);
        Assert.Equal(10.0, aggregated["bn1_gamma"][0], precision: 10);
        Assert.Equal(10.0, aggregated["bn1_gamma"][1], precision: 10);

        Assert.Equal("FedBN", aggregator.GetStrategyName());
        Assert.Contains("bn", aggregator.GetBatchNormPatterns());
    }
}

