namespace AiDotNet.Tests.FederatedLearning;

using Xunit;

public class AggregationStrategyBaseTests
{
    [Fact]
    public void GetTotalWeightOrThrow_Throws_WhenWeightsNull()
    {
        Assert.Throws<ArgumentException>(() =>
            AggregationStrategyBaseAccessor.CallGetTotalWeightOrThrow(
                clientWeights: null!,
                clientIds: new[] { 1 },
                paramName: "weights"));
    }

    [Fact]
    public void GetTotalWeightOrThrow_Throws_WhenWeightsEmpty()
    {
        Assert.Throws<ArgumentException>(() =>
            AggregationStrategyBaseAccessor.CallGetTotalWeightOrThrow(
                clientWeights: new Dictionary<int, double>(),
                clientIds: new[] { 1 },
                paramName: "weights"));
    }

    [Fact]
    public void GetTotalWeightOrThrow_Throws_WhenClientWeightMissing()
    {
        var weights = new Dictionary<int, double> { [1] = 1.0 };

        Assert.Throws<ArgumentException>(() =>
            AggregationStrategyBaseAccessor.CallGetTotalWeightOrThrow(
                clientWeights: weights,
                clientIds: new[] { 1, 2 },
                paramName: "weights"));
    }

    [Fact]
    public void GetTotalWeightOrThrow_Throws_WhenTotalWeightNotPositive()
    {
        var weights = new Dictionary<int, double> { [1] = 0.0, [2] = -1.0 };

        Assert.Throws<ArgumentException>(() =>
            AggregationStrategyBaseAccessor.CallGetTotalWeightOrThrow(
                clientWeights: weights,
                clientIds: new[] { 1, 2 },
                paramName: "weights"));
    }

    [Fact]
    public void GetTotalWeightOrThrow_ReturnsSum_ForValidWeights()
    {
        var weights = new Dictionary<int, double> { [1] = 0.25, [2] = 0.75 };

        var total = AggregationStrategyBaseAccessor.CallGetTotalWeightOrThrow(
            clientWeights: weights,
            clientIds: new[] { 1, 2 },
            paramName: "weights");

        Assert.Equal(1.0, total, precision: 10);
    }
}

