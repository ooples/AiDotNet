using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a metric passed to ConfigureRegressionMetric is actually computed and surfaced.
/// </summary>
/// <remarks>
/// The configured metric was stored in a field nothing read, so it was never computed and the
/// caller had no way to notice: the built-in ErrorStats/PredictionStats are reported for every run
/// regardless, so the output looked populated whether or not the requested metric ran.
/// </remarks>
public class ConfiguredMetricTests
{
    /// <summary>A metric with an unmistakable value, so only real computation can produce it.</summary>
    private sealed class SentinelRegressionMetric : IRegressionMetric<double>
    {
        public const double Sentinel = 12345.0;

        public bool WasComputed { get; private set; }

        public string Name => "sentinel-metric";

        public string Category => "regression";

        public string Description => "Returns a fixed sentinel so the test can prove it ran.";

        public MetricDirection Direction => MetricDirection.LowerIsBetter;

        public double MinValue => double.MinValue;

        public double MaxValue => double.MaxValue;

        public bool RequiresProbabilities => false;

        public bool SupportsMultiClass => false;

        public double Compute(ReadOnlySpan<double> predictions, ReadOnlySpan<double> actuals)
        {
            WasComputed = true;
            return Sentinel;
        }

        public MetricWithCI<double> ComputeWithCI(
            ReadOnlySpan<double> predictions,
            ReadOnlySpan<double> actuals,
            ConfidenceIntervalMethod method = ConfidenceIntervalMethod.PercentileBootstrap,
            double confidenceLevel = 0.95,
            int bootstrapSamples = 1000,
            int? randomSeed = null)
            => new() { Value = Sentinel, LowerBound = Sentinel, UpperBound = Sentinel };
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
    public async Task ConfiguredRegressionMetric_IsComputedAndSurfaced()
    {
        var (x, y) = BuildData();
        var metric = new SentinelRegressionMetric();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureRegressionMetric(metric)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        // Before the fix the metric was never invoked at all.
        Assert.True(metric.WasComputed, "the configured metric was never computed");
        Assert.True(
            result.ConfiguredMetrics.ContainsKey("sentinel-metric"),
            "the configured metric's value never reached AiModelResult.ConfiguredMetrics");
        Assert.Equal(SentinelRegressionMetric.Sentinel, result.ConfiguredMetrics["sentinel-metric"]);
    }

    [Fact(Timeout = 120000)]
    public async Task NoConfiguredMetric_LeavesTheCollectionEmpty()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Empty(result.ConfiguredMetrics);
    }
}
