using AiDotNet.Configuration;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>Pins adapter defects found by adversarial review of program-evolution configuration.</summary>
public sealed class EvolutionAuditRegressionTests
{
    [Fact]
    public void TwoAggregationRulesThatScoreDifferentlyCannotShareAVersionHash()
    {
        var mean = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Mean };
        var weighted = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Weighted,
            Weights = { ["accuracy"] = 1.0 }
        };
        var differentWeights = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Weighted,
            Weights = { ["latencyMs"] = 1.0 }
        };

        Assert.NotEqual(mean.ToString(), weighted.ToString());
        Assert.NotEqual(weighted.ToString(), differentWeights.ToString());

        var oneOrder = new ProgramMetricAggregationOptions { Weights = { ["a"] = 1, ["b"] = 2 } };
        var otherOrder = new ProgramMetricAggregationOptions { Weights = { ["b"] = 2, ["a"] = 1 } };
        Assert.Equal(oneOrder.ToString(), otherOrder.ToString());
    }

    [Fact]
    public void TheWeightedStrategiesCanBeExpressedByAConfigurationFile()
    {
        const string yaml = @"
programEvolution:
  metrics:
    strategy: Weighted
    weights:
      accuracy: 2.0
      latencyMs: 0.5
    excludedFeatureDimensions:
      - length
";
        YamlModelConfig config = YamlConfigLoader.LoadFromString(yaml);
        ProgramEvolutionOptions options = Assert.IsType<ProgramEvolutionOptions>(config.ProgramEvolution);

        Assert.Equal(ProgramMetricAggregationStrategy.Weighted, options.Metrics.Strategy);
        Assert.Equal(2.0, options.Metrics.Weights["accuracy"]);
        Assert.Equal(0.5, options.Metrics.Weights["latencyMs"]);
        Assert.Contains("length", options.Metrics.ExcludedFeatureDimensions);
        options.Metrics.Validate();
    }
}
