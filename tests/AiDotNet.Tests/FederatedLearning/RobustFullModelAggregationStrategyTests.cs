using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class RobustFullModelAggregationStrategyTests
{
    [Fact]
    public void Median_Aggregate_IgnoresOutlier()
    {
        var aggregator = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var models = CreateClientModels(new[] { 0.0, 0.0, 100.0, 0.0, 0.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, aggregated.GetParameters()[0], 12);
    }

    [Fact]
    public void TrimmedMean_Aggregate_IgnoresOutlier()
    {
        var aggregator = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, aggregated.GetParameters()[0], 12);
    }

    [Fact]
    public void Krum_Aggregate_SelectsCentralClient()
    {
        var aggregator = new KrumFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, aggregated.GetParameters()[0], 12);
    }

    [Fact]
    public void MultiKrum_Aggregate_AveragesSelectedCentralClients()
    {
        var aggregator = new MultiKrumFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(
            byzantineClientCount: 1,
            selectionCount: 3,
            useClientWeightsForAveraging: false);

        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 0.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(aggregated.GetParameters()[0] >= 0.0 && aggregated.GetParameters()[0] <= 1.0);
    }

    [Fact]
    public void Bulyan_Aggregate_IgnoresExtremeOutlier()
    {
        var aggregator = new BulyanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(aggregated.GetParameters()[0] >= 0.0 && aggregated.GetParameters()[0] <= 1.0);
    }

    [Fact]
    public void WinsorizedMean_Aggregate_ClipsOutlier()
    {
        var aggregator = new WinsorizedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(winsorizeFraction: 0.2);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(aggregated.GetParameters()[0] >= 0.0 && aggregated.GetParameters()[0] < 100.0);
    }

    [Fact]
    public void Rfa_Aggregate_IsRobustToOutlier()
    {
        var aggregator = new RfaFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(maxIterations: 5);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 1.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(aggregated.GetParameters()[0] >= 0.0 && aggregated.GetParameters()[0] <= 1.0);
    }

    private static Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>> CreateClientModels(double[] firstParameterValues)
    {
        var models = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < firstParameterValues.Length; i++)
        {
            int clientId = i + 1;
            var model = new MockFullModel(_ => new Vector<double>(1), parameterCount: 1);
            model.SetParameters(new Vector<double>(new[] { firstParameterValues[i] }));
            models[clientId] = model;
        }

        return models;
    }

    private static Dictionary<int, double> CreateEqualWeights(IEnumerable<int> clientIds)
    {
        var weights = new Dictionary<int, double>();
        foreach (var id in clientIds)
        {
            weights[id] = 1.0;
        }

        return weights;
    }
}
