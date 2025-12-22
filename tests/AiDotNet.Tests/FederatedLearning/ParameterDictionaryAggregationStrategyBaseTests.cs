using AiDotNet.FederatedLearning.Aggregators;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public sealed class ParameterDictionaryAggregationStrategyBaseTests : ParameterDictionaryAggregationStrategyBase<double>
{
    public override Dictionary<string, double[]> Aggregate(
        Dictionary<int, Dictionary<string, double[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    public override string GetStrategyName() => "TestParameterDictionaryBase";

    [Fact]
    public void AggregateWeightedAverage_ValidatesInputs()
    {
        Assert.Throws<ArgumentException>(() => AggregateWeightedAverage(null!, new Dictionary<int, double>()));
        Assert.Throws<ArgumentException>(() => AggregateWeightedAverage(new Dictionary<int, Dictionary<string, double[]>>(), new Dictionary<int, double>()));
        Assert.Throws<ArgumentException>(() => AggregateWeightedAverage(new Dictionary<int, Dictionary<string, double[]>> { [1] = new Dictionary<string, double[]>() }, null!));
    }

    [Fact]
    public void AggregateWeightedAverage_ThrowsWhenClientMissingLayerOrLengthMismatch()
    {
        var reference = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 1.0, 2.0 },
            ["b"] = new[] { 3.0 }
        };

        var modelsMissingLayer = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = reference,
            [2] = new Dictionary<string, double[]> { ["a"] = new[] { 1.0, 2.0 } }
        };

        Assert.Throws<ArgumentException>(() => AggregateWeightedAverage(modelsMissingLayer, new Dictionary<int, double> { [1] = 1.0, [2] = 1.0 }));

        var modelsLengthMismatch = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = reference,
            [2] = new Dictionary<string, double[]>
            {
                ["a"] = new[] { 1.0, 2.0, 3.0 },
                ["b"] = new[] { 3.0 }
            }
        };

        Assert.Throws<ArgumentException>(() => AggregateWeightedAverage(modelsLengthMismatch, new Dictionary<int, double> { [1] = 1.0, [2] = 1.0 }));
    }

    [Fact]
    public void AggregateWeightedAverage_ComputesWeightedAverage()
    {
        var client1 = new Dictionary<string, double[]>
        {
            ["w"] = new[] { 1.0, 3.0 }
        };
        var client2 = new Dictionary<string, double[]>
        {
            ["w"] = new[] { 5.0, 7.0 }
        };

        var models = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = client1,
            [2] = client2
        };

        var weights = new Dictionary<int, double>
        {
            [1] = 1.0,
            [2] = 3.0
        };

        var aggregated = AggregateWeightedAverage(models, weights);

        Assert.Single(aggregated);
        Assert.Equal((1.0 * 1.0 + 5.0 * 3.0) / 4.0, aggregated["w"][0], precision: 12);
        Assert.Equal((3.0 * 1.0 + 7.0 * 3.0) / 4.0, aggregated["w"][1], precision: 12);
    }

    [Fact]
    public void AggregateLayerWeightedAverageInto_UpdatesDestination()
    {
        var client1 = new Dictionary<string, double[]> { ["w"] = new[] { 1.0, 3.0 } };
        var client2 = new Dictionary<string, double[]> { ["w"] = new[] { 5.0, 7.0 } };

        var models = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = client1,
            [2] = client2
        };
        var weights = new Dictionary<int, double> { [1] = 1.0, [2] = 3.0 };

        var destination = CreateZeroInitializedLayer(length: 2);
        AggregateLayerWeightedAverageInto(
            layerName: "w",
            clientModels: models,
            clientWeights: weights,
            totalWeight: 4.0,
            destination: destination);

        Assert.Equal((1.0 * 1.0 + 5.0 * 3.0) / 4.0, destination[0], precision: 12);
        Assert.Equal((3.0 * 1.0 + 7.0 * 3.0) / 4.0, destination[1], precision: 12);
    }

    [Fact]
    public void AggregateLayerWeightedAverageInto_ThrowsForNullDestination()
    {
        Assert.Throws<ArgumentNullException>(() => AggregateLayerWeightedAverageInto(
            layerName: "w",
            clientModels: new Dictionary<int, Dictionary<string, double[]>>(),
            clientWeights: new Dictionary<int, double>(),
            totalWeight: 1.0,
            destination: null!));
    }
}
