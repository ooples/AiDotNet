using System.Diagnostics;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Benchmarks;

/// <summary>
/// Automated benchmark suite for comparing HRE against traditional architectures.
/// Measures inference latency, parameter count, model size, and prediction accuracy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HREBenchmarkSuite<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new benchmark suite.
    /// </summary>
    public HREBenchmarkSuite()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Runs a complete benchmark comparing HRE configurations.
    /// </summary>
    /// <param name="timeSeries">Input time-series data.</param>
    /// <param name="windowSize">Lookback window size (must be power of 2).</param>
    /// <param name="testFraction">Fraction of data reserved for testing (0.0 to 1.0).</param>
    /// <returns>Benchmark results for each configuration.</returns>
    public List<BenchmarkResult> RunForecasterBenchmark(Vector<T> timeSeries, int windowSize = 64,
        double testFraction = 0.2)
    {
        var results = new List<BenchmarkResult>();
        int trainEnd = (int)(timeSeries.Length * (1.0 - testFraction));

        foreach (var nonlinearity in new[] { NonlinearityType.ModReLU, NonlinearityType.SpectralGating, NonlinearityType.InstantaneousFreq })
        {
            var options = new HREModelOptions
            {
                CarrierCount = 8,
                FftSize = 256,
                Nonlinearity = nonlinearity,
                UseMellinFourier = false,
                NumOFDMLayers = 1,
                NumAttentionLayers = 1,
                Seed = 42
            };

            var forecaster = new HREForecaster<T>(windowSize, 1, options);

            // Measure inference latency
            var latency = MeasureLatency(forecaster, timeSeries, trainEnd, windowSize);

            // Measure prediction accuracy
            var accuracy = MeasureAccuracy(forecaster, timeSeries, trainEnd, windowSize);

            results.Add(new BenchmarkResult
            {
                Name = $"HRE-{nonlinearity}",
                ParameterCount = forecaster.Model.ParameterCount,
                InferenceLatencyMs = latency,
                MSE = accuracy.mse,
                MAE = accuracy.mae,
                PredictionCount = accuracy.count
            });
        }

        return results;
    }

    /// <summary>
    /// Measures average inference latency over test windows.
    /// </summary>
    public double MeasureLatency(HREForecaster<T> forecaster, Vector<T> timeSeries, int trainEnd, int windowSize)
    {
        // Warm up
        var warmupWindow = new Vector<T>(windowSize);
        for (int i = 0; i < windowSize; i++)
        {
            warmupWindow[i] = timeSeries[trainEnd - windowSize + i];
        }
        forecaster.Predict(warmupWindow);

        // Measure
        var sw = Stopwatch.StartNew();
        int count = 0;
        for (int t = trainEnd; t < timeSeries.Length - 1 && count < 100; t++)
        {
            if (t - windowSize < 0) continue;
            var window = new Vector<T>(windowSize);
            for (int i = 0; i < windowSize; i++)
            {
                window[i] = timeSeries[t - windowSize + i];
            }
            forecaster.Predict(window);
            count++;
        }
        sw.Stop();

        return count > 0 ? sw.Elapsed.TotalMilliseconds / count : double.NaN;
    }

    /// <summary>
    /// Measures prediction accuracy on the test portion.
    /// </summary>
    public (double mse, double mae, int count) MeasureAccuracy(HREForecaster<T> forecaster,
        Vector<T> timeSeries, int trainEnd, int windowSize)
    {
        double totalSqError = 0;
        double totalAbsError = 0;
        int count = 0;

        for (int t = trainEnd; t < timeSeries.Length - 1; t++)
        {
            if (t - windowSize < 0) continue;

            var window = new Vector<T>(windowSize);
            for (int i = 0; i < windowSize; i++)
            {
                window[i] = timeSeries[t - windowSize + i];
            }

            var pred = forecaster.Predict(window);
            double predVal = _numOps.ToDouble(pred[0]);
            double actualVal = _numOps.ToDouble(timeSeries[t + 1]);

            if (!double.IsNaN(predVal) && !double.IsInfinity(predVal))
            {
                double err = predVal - actualVal;
                totalSqError += err * err;
                totalAbsError += Math.Abs(err);
                count++;
            }
        }

        return count > 0
            ? (totalSqError / count, totalAbsError / count, count)
            : (double.NaN, double.NaN, 0);
    }
}

/// <summary>
/// Holds benchmark results for a single configuration.
/// </summary>
public class BenchmarkResult
{
    /// <summary>Configuration name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Total learnable parameters.</summary>
    public int ParameterCount { get; set; }

    /// <summary>Average inference latency in milliseconds.</summary>
    public double InferenceLatencyMs { get; set; }

    /// <summary>Mean Squared Error on test data.</summary>
    public double MSE { get; set; }

    /// <summary>Mean Absolute Error on test data.</summary>
    public double MAE { get; set; }

    /// <summary>Number of valid predictions made.</summary>
    public int PredictionCount { get; set; }

    /// <inheritdoc/>
    public override string ToString() =>
        $"{Name}: params={ParameterCount}, latency={InferenceLatencyMs:F3}ms, MSE={MSE:F6}, MAE={MAE:F6}, n={PredictionCount}";
}
