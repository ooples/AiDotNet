using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FedProxFullModelAggregationStrategyTests
{
    [Fact]
    public void Ctor_WithNegativeMu_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FedProxFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(-0.1));
    }

    [Fact]
    public void GetMu_ReturnsConfiguredValue()
    {
        var strategy = new FedProxFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(mu: 0.5);
        Assert.Equal(0.5, strategy.GetMu(), precision: 10);
    }

    [Fact]
    public void GetStrategyName_ReturnsFedProx()
    {
        var strategy = new FedProxFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        Assert.Equal("FedProx", strategy.GetStrategyName());
    }

    [Fact]
    public void Aggregate_DelegatesToFedAvg()
    {
        var model0 = new MockFullModel(_ => new Vector<double>(3), parameterCount: 3);
        model0.SetParameters(new Vector<double>(new[] { 1.0, 1.0, 1.0 }));

        var model1 = new MockFullModel(_ => new Vector<double>(3), parameterCount: 3);
        model1.SetParameters(new Vector<double>(new[] { 3.0, 3.0, 3.0 }));

        var strategy = new FedProxFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(mu: 0.0);
        var aggregated = strategy.Aggregate(
            new Dictionary<int, AiDotNet.Interfaces.IFullModel<double, Matrix<double>, Vector<double>>>
            {
                [0] = model0,
                [1] = model1
            },
            new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0
            });

        var p = aggregated.GetParameters();
        Assert.Equal(2.0, p[0], precision: 10);
        Assert.Equal(2.0, p[1], precision: 10);
        Assert.Equal(2.0, p[2], precision: 10);
    }
}

