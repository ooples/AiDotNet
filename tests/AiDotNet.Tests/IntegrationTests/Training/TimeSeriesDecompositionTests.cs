using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureTimeSeriesDecomposition runs a real post-build audit: it re-decomposes the model's
/// target series, grades trend/seasonal strength and residual whiteness, checks additive reconstruction, and
/// compares decomposition-based forecasting against a random-walk baseline — surfaced on AiModelResult.
/// </summary>
public class TimeSeriesDecompositionTests
{
    private const int Period = 4;

    /// <summary>
    /// A minimal additive decomposition (trend = centered moving average, seasonal = per-phase detrended mean,
    /// residual = remainder) with a single-argument constructor so the audit can re-decompose a new series.
    /// </summary>
    private sealed class TestAdditiveDecomposition : ITimeSeriesDecomposition<double>
    {
        private readonly Dictionary<DecompositionComponentType, object> _components = new();

        public TestAdditiveDecomposition(Vector<double> timeSeries)
        {
            TimeSeries = timeSeries;
            int n = timeSeries.Length;
            var s = new double[n];
            for (int i = 0; i < n; i++) s[i] = timeSeries[i];

            // Trend: centered moving average over one period (clamped at the edges).
            var trend = new double[n];
            int h = Period / 2;
            for (int i = 0; i < n; i++)
            {
                double sum = 0; int cnt = 0;
                for (int j = Math.Max(0, i - h); j <= Math.Min(n - 1, i + h); j++) { sum += s[j]; cnt++; }
                trend[i] = sum / cnt;
            }

            // Seasonal: average of the detrended series by phase, centered to mean zero.
            var phaseSum = new double[Period];
            var phaseCnt = new int[Period];
            for (int i = 0; i < n; i++) { int p = i % Period; phaseSum[p] += s[i] - trend[i]; phaseCnt[p]++; }
            var phaseMean = new double[Period];
            double gMean = 0;
            for (int p = 0; p < Period; p++) { phaseMean[p] = phaseCnt[p] > 0 ? phaseSum[p] / phaseCnt[p] : 0; gMean += phaseMean[p]; }
            gMean /= Period;
            var seasonal = new double[n];
            for (int i = 0; i < n; i++) seasonal[i] = phaseMean[i % Period] - gMean;

            var residual = new double[n];
            for (int i = 0; i < n; i++) residual[i] = s[i] - trend[i] - seasonal[i];

            _components[DecompositionComponentType.Trend] = ToVector(trend);
            _components[DecompositionComponentType.Seasonal] = ToVector(seasonal);
            _components[DecompositionComponentType.Residual] = ToVector(residual);
        }

        private static Vector<double> ToVector(double[] a)
        {
            var v = new Vector<double>(a.Length);
            for (int i = 0; i < a.Length; i++) v[i] = a[i];
            return v;
        }

        public Vector<double> TimeSeries { get; }
        public Dictionary<DecompositionComponentType, object> GetComponents() => _components;
        public object? GetComponent(DecompositionComponentType componentType)
            => _components.TryGetValue(componentType, out var c) ? c : null;
        public bool HasComponent(DecompositionComponentType componentType) => _components.ContainsKey(componentType);
    }

    private static (Matrix<double> X, Vector<double> Y) BuildSeasonalData(int rows = 96, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2) + (i * 0.01);
            // Clear upward trend plus a period-4 seasonal wave plus a touch of noise.
            double seasonal = (i % Period == 0) ? 2.0 : (i % Period == 2 ? -2.0 : 0.0);
            y[i] = 0.15 * i + seasonal + 0.05 * Math.Sin(i * 0.9);
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTimeSeriesDecomposition_GradesDecompositionAndForecastValue()
    {
        var (x, y) = BuildSeasonalData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureTimeSeriesDecomposition(new TestAdditiveDecomposition(new Vector<double>(4)))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        var report = result.TimeSeriesDecomposition;
        Assert.NotNull(report);
        Assert.Equal(nameof(TestAdditiveDecomposition), report?.MethodName);

        // The audit re-decomposed the model's own target series (single-arg ctor re-instantiation succeeded).
        Assert.Equal(DecompositionAnalysisSource.TargetSeries, report?.AnalyzedSeries);
        Assert.Contains(DecompositionComponentType.Trend, report?.ComponentsPresent ?? new List<DecompositionComponentType>());
        Assert.Contains(DecompositionComponentType.Seasonal, report?.ComponentsPresent ?? new List<DecompositionComponentType>());

        // A trending, seasonal series should grade as strongly trended and seasonal.
        Assert.True((report?.TrendStrength ?? 0) > 0.5, $"trend strength {report?.TrendStrength}");
        Assert.True((report?.SeasonalStrength ?? 0) > 0.3, $"seasonal strength {report?.SeasonalStrength}");

        // Additive reconstruction (trend + seasonal + residual) should return the series almost exactly.
        Assert.True(report?.ReconstructionAvailable == true);
        Assert.True((report?.ReconstructionError ?? 1.0) < 1e-6, $"reconstruction error {report?.ReconstructionError}");

        // Decomposition-based forecasting should beat a random-walk baseline on a trending, seasonal series.
        Assert.True(report?.ForecastEvaluated == true);
        Assert.True((report?.NaiveForecastRmse ?? 0) > 0);
        Assert.True((report?.ForecastSkill ?? -1) > 0, $"forecast skill {report?.ForecastSkill}");
    }

    [Fact(Timeout = 120000)]
    public async Task NoDecomposition_LeavesReportNull()
    {
        var (x, y) = BuildSeasonalData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.TimeSeriesDecomposition);
    }
}
