using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

public class RobustFullModelAggregationStrategyTests
{
    [Fact(Timeout = 60000)]
    public async Task Median_Aggregate_IgnoresOutlier()
    {
        var aggregator = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var models = CreateClientModels(new[] { 0.0, 0.0, 100.0, 0.0, 0.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task TrimmedMean_Aggregate_IgnoresOutlier()
    {
        var aggregator = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task Krum_Aggregate_SelectsCentralClient()
    {
        var aggregator = new KrumFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.Equal(0.0, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task MultiKrum_Aggregate_AveragesSelectedCentralClients()
    {
        var aggregator = new MultiKrumFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(
            byzantineClientCount: 1,
            selectionCount: 3,
            useClientWeightsForAveraging: false);

        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 0.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] >= 0.0 && ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] <= 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task Bulyan_Aggregate_IgnoresExtremeOutlier()
    {
        var aggregator = new BulyanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] >= 0.0 && ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] <= 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task WinsorizedMean_Aggregate_ClipsOutlier()
    {
        var aggregator = new WinsorizedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(winsorizeFraction: 0.2);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] >= 0.0 && ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] < 100.0);
    }

    [Fact(Timeout = 60000)]
    public async Task Rfa_Aggregate_IsRobustToOutlier()
    {
        var aggregator = new RfaFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(maxIterations: 5);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 1.0, 100.0 });
        var weights = CreateEqualWeights(models.Keys);

        var aggregated = aggregator.Aggregate(models, weights);
        Assert.True(((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] >= 0.0 && ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0] <= 1.0);
    }

    // A Byzantine client can send any value, including +/-Infinity and NaN. The order-statistic aggregators
    // drop (or clamp) the extreme order statistics, so a single non-finite value must not reach the result.
    // The expected values are those of the per-coordinate Array.Sort loops these aggregators replaced, where
    // NaN sorts below every number and so lands in the trimmed lower tail.

    [Theory(Timeout = 60000)]
    [InlineData(double.PositiveInfinity, 3.0)] // sorted 1 2 3 4 +Inf -> mean(2, 3, 4)
    [InlineData(double.NegativeInfinity, 2.0)] // sorted -Inf 1 2 3 4 -> mean(1, 2, 3)
    [InlineData(double.NaN, 2.0)]              // sorted NaN 1 2 3 4 -> mean(1, 2, 3)
    public async Task TrimmedMean_Aggregate_DropsNonFiniteOutlier(double outlier, double expected)
    {
        await Task.Yield();

        var aggregator = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var models = CreateClientModels(new[] { 1.0, 2.0, outlier, 3.0, 4.0 });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));
        Assert.Equal(expected, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Theory(Timeout = 60000)]
    [InlineData(double.PositiveInfinity, 2.5)] // sorted 1 2 3 +Inf -> (2 + 3) / 2
    [InlineData(double.NegativeInfinity, 1.5)] // sorted -Inf 1 2 3 -> (1 + 2) / 2
    [InlineData(double.NaN, 1.5)]              // sorted NaN 1 2 3 -> (1 + 2) / 2
    public async Task Median_Aggregate_EvenClientCount_DropsNonFiniteOutlier(double outlier, double expected)
    {
        await Task.Yield();

        var aggregator = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var models = CreateClientModels(new[] { 1.0, outlier, 2.0, 3.0 });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));
        Assert.Equal(expected, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Theory(Timeout = 60000)]
    [InlineData(double.PositiveInfinity, 3.0)] // sorted 1 2 3 4 +Inf -> 3
    [InlineData(double.NaN, 2.0)]              // sorted NaN 1 2 3 4 -> 2
    public async Task Median_Aggregate_OddClientCount_DropsNonFiniteOutlier(double outlier, double expected)
    {
        await Task.Yield();

        var aggregator = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var models = CreateClientModels(new[] { 1.0, 2.0, outlier, 3.0, 4.0 });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));
        Assert.Equal(expected, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Theory(Timeout = 60000)]
    [InlineData(double.PositiveInfinity, 3.0)] // sorted 1 2 3 4 +Inf, clamped to [2, 4] -> (2 + 2 + 3 + 4 + 4) / 5
    [InlineData(double.NegativeInfinity, 2.0)] // sorted -Inf 1 2 3 4, clamped to [1, 3] -> (1 + 1 + 2 + 3 + 3) / 5
    public async Task WinsorizedMean_Aggregate_ClampsInfiniteOutlier(double outlier, double expected)
    {
        await Task.Yield();

        var aggregator = new WinsorizedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(winsorizeFraction: 0.2);
        var models = CreateClientModels(new[] { 1.0, 2.0, outlier, 3.0, 4.0 });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));
        Assert.Equal(expected, ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0], 12);
    }

    [Theory(Timeout = 60000)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.NaN)]
    public async Task Bulyan_Aggregate_IgnoresNonFiniteOutlier(double outlier)
    {
        await Task.Yield();

        var aggregator = new BulyanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, outlier });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));
        double value = ((IParameterizable<double, Matrix<double>, Vector<double>>)aggregated).GetParameters()[0];
        Assert.True(value >= 0.0 && value <= 1.0, $"Bulyan returned {value}.");
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
