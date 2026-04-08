using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public sealed class RobustFullModelAggregationStrategyBaseTests
    : RobustFullModelAggregationStrategyBase<double, Matrix<double>, Vector<double>>
{
    public override IFullModel<double, Matrix<double>, Vector<double>> Aggregate(
        Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return clientModels.First().Value;
    }

    public override string GetStrategyName() => "TestRobustBase";

    [Fact]
    public void GetReferenceModelOrThrow_ThrowsForNullOrEmpty()
    {
        Assert.Throws<ArgumentException>(() => GetReferenceModelOrThrow(null!));
        Assert.Throws<ArgumentException>(() => GetReferenceModelOrThrow(new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>()));
    }

    [Fact]
    public void GetClientParametersOrThrow_ThrowsForParameterLengthMismatch()
    {
        var model1 = new MockFullModel(_ => new Vector<double>(1), parameterCount: 2);
        var model2 = new MockFullModel(_ => new Vector<double>(1), parameterCount: 3);

        var models = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            [1] = model1,
            [2] = model2
        };

        Assert.Throws<ArgumentException>(() => GetClientParametersOrThrow(models, expectedParameterCount: 2));
    }

    [Fact]
    public void ComputeSquaredL2Distance_ThrowsForLengthMismatch()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.Throws<ArgumentException>(() => ComputeSquaredL2Distance(a, b));
    }

    [Fact]
    public void WeightedAverageOrUnweightedAverage_ComputesExpectedValues()
    {
        var selected = new List<int> { 1, 2 };

        var parameters = new Dictionary<int, Vector<double>>
        {
            [1] = new Vector<double>(new[] { 1.0, 3.0 }),
            [2] = new Vector<double>(new[] { 5.0, 7.0 })
        };

        var unweighted = WeightedAverageOrUnweightedAverage(
            selectedClientIds: selected,
            clientParameters: parameters,
            clientWeights: new Dictionary<int, double>(),
            useClientWeights: false);

        Assert.Equal(3.0, unweighted[0], precision: 12);
        Assert.Equal(5.0, unweighted[1], precision: 12);

        var weights = new Dictionary<int, double> { [1] = 1.0, [2] = 3.0 };
        var weighted = WeightedAverageOrUnweightedAverage(
            selectedClientIds: selected,
            clientParameters: parameters,
            clientWeights: weights,
            useClientWeights: true);

        Assert.Equal((1.0 * 1.0 + 5.0 * 3.0) / 4.0, weighted[0], precision: 12);
        Assert.Equal((3.0 * 1.0 + 7.0 * 3.0) / 4.0, weighted[1], precision: 12);
    }

    [Fact]
    public void WeightedAverageOrUnweightedAverage_ThrowsWhenClientsMissing()
    {
        Assert.Throws<ArgumentException>(() => WeightedAverageOrUnweightedAverage(
            selectedClientIds: null!,
            clientParameters: new Dictionary<int, Vector<double>>(),
            clientWeights: new Dictionary<int, double>(),
            useClientWeights: false));

        Assert.Throws<ArgumentException>(() => WeightedAverageOrUnweightedAverage(
            selectedClientIds: new List<int>(),
            clientParameters: new Dictionary<int, Vector<double>>(),
            clientWeights: new Dictionary<int, double>(),
            useClientWeights: false));

        Assert.Throws<ArgumentException>(() => WeightedAverageOrUnweightedAverage(
            selectedClientIds: new List<int> { 1 },
            clientParameters: new Dictionary<int, Vector<double>>(),
            clientWeights: new Dictionary<int, double>(),
            useClientWeights: false));
    }
}
