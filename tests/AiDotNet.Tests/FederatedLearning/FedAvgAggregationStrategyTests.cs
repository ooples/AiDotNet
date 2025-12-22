namespace AiDotNet.Tests.FederatedLearning;

using System;
using System.Collections.Generic;
using AiDotNet.FederatedLearning.Aggregators;
using Xunit;

/// <summary>
/// Unit tests for FedAvg (Federated Averaging) aggregation strategy.
/// </summary>
public class FedAvgAggregationStrategyTests
{
    [Fact]
    public void Aggregate_WithEqualWeights_ReturnsAverageModel()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();

        // Create two client models with simple parameters
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 1.0, 2.0, 3.0 }
            },
            [1] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 3.0, 4.0, 5.0 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0
        };

        // Act
        var aggregatedModel = strategy.Aggregate(clientModels, clientWeights);

        // Assert
        Assert.NotNull(aggregatedModel);
        Assert.Contains("layer1", aggregatedModel.Keys);
        Assert.Equal(3, aggregatedModel["layer1"].Length);

        // Expected: (1+3)/2=2, (2+4)/2=3, (3+5)/2=4
        Assert.Equal(2.0, aggregatedModel["layer1"][0], precision: 5);
        Assert.Equal(3.0, aggregatedModel["layer1"][1], precision: 5);
        Assert.Equal(4.0, aggregatedModel["layer1"][2], precision: 5);
    }

    [Fact]
    public void Aggregate_WithDifferentWeights_ReturnsWeightedAverage()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 1.0, 2.0 }
            },
            [1] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 3.0, 4.0 }
            }
        };

        // Client 1 has 3x the weight (3x more data)
        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 3.0
        };

        // Act
        var aggregatedModel = strategy.Aggregate(clientModels, clientWeights);

        // Assert
        // Expected: (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        //           (2*1 + 4*3)/(1+3) = 14/4 = 3.5
        Assert.Equal(2.5, aggregatedModel["layer1"][0], precision: 5);
        Assert.Equal(3.5, aggregatedModel["layer1"][1], precision: 5);
    }

    [Fact]
    public void Aggregate_WithMultipleLayers_AggregatesAllLayers()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 1.0 },
                ["layer2"] = new double[] { 2.0 }
            },
            [1] = new Dictionary<string, double[]>
            {
                ["layer1"] = new double[] { 3.0 },
                ["layer2"] = new double[] { 4.0 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0
        };

        // Act
        var aggregatedModel = strategy.Aggregate(clientModels, clientWeights);

        // Assert
        Assert.Equal(2, aggregatedModel.Count);
        Assert.Contains("layer1", aggregatedModel.Keys);
        Assert.Contains("layer2", aggregatedModel.Keys);
        Assert.Equal(2.0, aggregatedModel["layer1"][0], precision: 5);
        Assert.Equal(3.0, aggregatedModel["layer2"][0], precision: 5);
    }

    [Fact]
    public void Aggregate_WithEmptyClientModels_ThrowsArgumentException()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();
        var emptyModels = new Dictionary<int, Dictionary<string, double[]>>();
        var clientWeights = new Dictionary<int, double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.Aggregate(emptyModels, clientWeights));
    }

    [Fact]
    public void Aggregate_WithNullClientModels_ThrowsArgumentException()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();
        Dictionary<int, Dictionary<string, double[]>>? nullModels = null;
        var clientWeights = new Dictionary<int, double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.Aggregate(nullModels!, clientWeights));
    }

    [Fact]
    public void GetStrategyName_ReturnsCorrectName()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();

        // Act
        var name = strategy.GetStrategyName();

        // Assert
        Assert.Equal("FedAvg", name);
    }

    [Fact]
    public void Aggregate_WithThreeClients_ComputesCorrectWeightedAverage()
    {
        // Arrange
        var strategy = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]>
            {
                ["weights"] = new double[] { 0.1, 0.2, 0.3 }
            },
            [1] = new Dictionary<string, double[]>
            {
                ["weights"] = new double[] { 0.2, 0.3, 0.4 }
            },
            [2] = new Dictionary<string, double[]>
            {
                ["weights"] = new double[] { 0.3, 0.4, 0.5 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 100.0,  // 100 samples
            [1] = 200.0,  // 200 samples
            [2] = 300.0   // 300 samples
        };

        // Act
        var aggregatedModel = strategy.Aggregate(clientModels, clientWeights);

        // Assert
        // Expected: (0.1*100 + 0.2*200 + 0.3*300) / 600 = (10 + 40 + 90) / 600 = 140/600 = 0.2333...
        //           (0.2*100 + 0.3*200 + 0.4*300) / 600 = (20 + 60 + 120) / 600 = 200/600 = 0.3333...
        //           (0.3*100 + 0.4*200 + 0.5*300) / 600 = (30 + 80 + 150) / 600 = 260/600 = 0.4333...
        Assert.Equal(0.2333333, aggregatedModel["weights"][0], precision: 5);
        Assert.Equal(0.3333333, aggregatedModel["weights"][1], precision: 5);
        Assert.Equal(0.4333333, aggregatedModel["weights"][2], precision: 5);
    }
}
