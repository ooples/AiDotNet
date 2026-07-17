using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureDistanceMetric and ConfigureSimilarityMetric are actually computed against the
/// model's test-partition predictions and surfaced on AiModelResult.ConfiguredMetrics.
/// </summary>
/// <remarks>
/// Both stored a private field nothing read, so the configured metric never ran. They now auto-evaluate
/// alongside the regression/classification metrics: the configured distance/similarity between the
/// prediction and target vectors is reported. Sentinel implementations prove the path executed.
/// </remarks>
public class ConfiguredDistanceSimilarityMetricTests
{
    private sealed class SentinelDistanceMetric : DistanceMetricBase<double>
    {
        public const double Sentinel = 424242.0;
        public bool WasComputed { get; private set; }
        public override string Name => "sentinel-distance";
        public override double Compute(Vector<double> a, Vector<double> b)
        {
            WasComputed = true;
            return Sentinel;
        }
    }

    private sealed class SentinelSimilarityMetric : ISimilarityMetric<double>
    {
        public const double Sentinel = 0.98765;
        public bool WasComputed { get; private set; }
        public bool HigherIsBetter => true;
        public double Calculate(Vector<double> v1, Vector<double> v2)
        {
            WasComputed = true;
            return Sentinel;
        }
    }

    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 120, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.01);
            y[i] = Math.Sin((i + cols) * 0.15) + (i * 0.01);
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredDistanceMetric_IsComputedAndSurfaced()
    {
        var (x, y) = BuildData();
        var metric = new SentinelDistanceMetric();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDistanceMetric(metric)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.True(metric.WasComputed, "the configured distance metric was never computed");
        Assert.True(
            result.ConfiguredMetrics.ContainsKey("sentinel-distance"),
            "the configured distance metric never reached AiModelResult.ConfiguredMetrics");
        Assert.Equal(SentinelDistanceMetric.Sentinel, result.ConfiguredMetrics["sentinel-distance"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredSimilarityMetric_IsComputedAndSurfaced()
    {
        var (x, y) = BuildData();
        var metric = new SentinelSimilarityMetric();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureSimilarityMetric(metric)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.True(metric.WasComputed, "the configured similarity metric was never computed");
        Assert.True(
            result.ConfiguredMetrics.ContainsKey(nameof(SentinelSimilarityMetric)),
            "the configured similarity metric never reached AiModelResult.ConfiguredMetrics");
        Assert.Equal(
            SentinelSimilarityMetric.Sentinel,
            result.ConfiguredMetrics[nameof(SentinelSimilarityMetric)]);
    }
}
