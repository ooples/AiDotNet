using System;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;
using System.Collections.Generic;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Reference tests for the centered moving average used by the fast STL decomposition.
/// </summary>
public class STLDecompositionMovingAverageTests
{
    private static Vector<double> MovingAverage(Vector<double> data, int window)
    {
        var model = new STLDecomposition<double>(new STLDecompositionOptions<double> { SeasonalPeriod = 2 });
        var method = typeof(STLDecomposition<double>).GetMethod("MovingAverage", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);
        var result = method?.Invoke(model, new object[] { data, window }) as Vector<double>;
        Assert.NotNull(result);
        return result ?? new Vector<double>(0);
    }

    /// <summary>
    /// Naive reference: interior points average the <paramref name="window"/> points starting at i - window / 2;
    /// the first and last window / 2 points average the available points of [i - window / 2, i + window / 2].
    /// </summary>
    private static double[] Reference(double[] data, int window)
    {
        int n = data.Length;
        int half = window / 2;
        var expected = new double[n];
        for (int i = 0; i < n; i++)
        {
            int start = i < half || i >= n - half ? Math.Max(0, i - half) : i - half;
            int end = i < half || i >= n - half ? Math.Min(n, i + half + 1) : i - half + window;
            double sum = 0;
            for (int j = start; j < end; j++)
            {
                sum += data[j];
            }

            expected[i] = sum / (end - start);
        }

        return expected;
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void MovingAverage_MatchesReferenceWindows(int window)
    {
        var data = new double[20];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (i * 7 % 11) + 0.25 * i;
        }

        var actual = MovingAverage(new Vector<double>(data), window);
        var expected = Reference(data, window);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 10);
        }
    }

    [Fact]
    public void MovingAverage_EvenWindow_IncludesEveryPointOnce()
    {
        // With an even window the running-sum update used to skip data[window] and later subtract it,
        // leaving every later interior average short by data[window] / window.
        var data = new double[12];
        data[4] = 8.0; // data[window] for window = 4
        var actual = MovingAverage(new Vector<double>(data), 4);

        // Interior windows [i - 2, i + 1] contain index 4 for i = 3..6 and no other point is non-zero.
        Assert.Equal(2.0, actual[3], 10);
        Assert.Equal(2.0, actual[4], 10);
        Assert.Equal(2.0, actual[5], 10);
        Assert.Equal(2.0, actual[6], 10);
        Assert.Equal(0.0, actual[7], 10);
        Assert.Equal(0.0, actual[9], 10);
    }
}

/// <summary>
/// Per-step sample quantiles (used by Chronos and DeepAR) must rank NaN samples exactly like master's
/// double sort (List&lt;double&gt;.Sort / OrderBy), which places NaN before every number, on every sort path.
/// </summary>
public class SampleQuantilesNaNTests
{
    private const double NaN = double.NaN;
    private const double Inf = double.PositiveInfinity;

    // Five sampled paths over three steps. Ranked per step, NaN first:
    // step 0: NaN, NaN, 0.5, 1, 2   step 1: NaN, NaN, -3, 2, 4   step 2: NaN, -1, 3, 5, +inf
    private static readonly double[][] MultiStepSamples =
    {
        new[] { 1.0, NaN, 3.0 },
        new[] { NaN, 2.0, NaN },
        new[] { 0.5, NaN, -1.0 },
        new[] { 2.0, 4.0, Inf },
        new[] { NaN, -3.0, 5.0 },
    };

    private static readonly double[] MultiStepQuantiles = { 0.0, 0.2, 0.5, 0.7, 0.9, 1.0 };

    private static readonly double[][] MultiStepExpected =
    {
        new[] { NaN, NaN, NaN },    // q = 0.0 -> rank 0
        new[] { NaN, NaN, -1.0 },   // q = 0.2 -> rank 1
        new[] { 0.5, -3.0, 3.0 },   // q = 0.5 -> rank 2
        new[] { 1.0, 2.0, 5.0 },    // q = 0.7 -> rank 3
        new[] { 2.0, 4.0, Inf },    // q = 0.9 -> rank 4
        new[] { 2.0, 4.0, Inf },    // q = 1.0 -> rank 5 clamped to 4
    };

    // A single step takes the engine's fast float sort path (one sorted axis, inner size 1)
    private static readonly double[][] SingleStepSamples =
    {
        new[] { 3.0 }, new[] { NaN }, new[] { 1.0 }, new[] { NaN }, new[] { 2.0 },
    };

    private static readonly double[] SingleStepQuantiles = { 0.0, 0.3, 0.4, 0.6, 0.99 };

    // Ranked: NaN, NaN, 1, 2, 3
    private static readonly double[][] SingleStepExpected =
    {
        new[] { NaN }, new[] { NaN }, new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 },
    };

    [Fact]
    public void MultiStep_Double_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<double>(MultiStepSamples, MultiStepQuantiles, MultiStepExpected, v => v, v => v);
    }

    [Fact]
    public void MultiStep_Float_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<float>(MultiStepSamples, MultiStepQuantiles, MultiStepExpected, v => (float)v, v => v);
    }

    [Fact]
    public void SingleStep_Double_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<double>(SingleStepSamples, SingleStepQuantiles, SingleStepExpected, v => v, v => v);
    }

    [Fact]
    public void SingleStep_Float_FastSortPath_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<float>(SingleStepSamples, SingleStepQuantiles, SingleStepExpected, v => (float)v, v => v);
    }

    [Fact]
    public void Expectations_MatchMastersDoubleSort()
    {
        // The hard-coded expectations are what master's List<double>.Sort-based pick returns
        AssertMatchesReference(MultiStepSamples, MultiStepQuantiles, MultiStepExpected);
        AssertMatchesReference(SingleStepSamples, SingleStepQuantiles, SingleStepExpected);
    }

    private static void AssertQuantiles<T>(
        double[][] samples, double[] quantiles, double[][] expected, Func<double, T> fromDouble, Func<T, double> toDouble)
    {
        var paths = new List<Vector<T>>();
        foreach (var sample in samples)
        {
            paths.Add(new Vector<T>(Array.ConvertAll(sample, v => fromDouble(v))));
        }

        int horizon = samples[0].Length;
        var result = TimeSeriesEngineOps<T>.SampleQuantiles(paths, quantiles, horizon);

        Assert.Equal(quantiles.Length, result.Count);
        for (int k = 0; k < quantiles.Length; k++)
        {
            var actual = result[quantiles[k]];
            Assert.Equal(horizon, actual.Length);
            for (int h = 0; h < horizon; h++)
            {
                double value = toDouble(actual[h]);
                if (double.IsNaN(expected[k][h]))
                {
                    Assert.True(double.IsNaN(value), $"q={quantiles[k]}, step {h}: expected NaN, got {value}");
                }
                else
                {
                    Assert.Equal(expected[k][h], value);
                }
            }
        }
    }

    private static void AssertMatchesReference(double[][] samples, double[] quantiles, double[][] expected)
    {
        int horizon = samples[0].Length;
        for (int k = 0; k < quantiles.Length; k++)
        {
            for (int h = 0; h < horizon; h++)
            {
                var values = new List<double>(samples.Length);
                foreach (var sample in samples)
                {
                    values.Add(sample[h]);
                }

                values.Sort();
                int idx = (int)(quantiles[k] * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                double reference = values[idx];
                if (double.IsNaN(reference))
                {
                    Assert.True(double.IsNaN(expected[k][h]));
                }
                else
                {
                    Assert.Equal(reference, expected[k][h]);
                }
            }
        }
    }
}
/// <summary>
/// Element counts behind the STL phase means and edge averages must be exact integers. A sum of ones in
/// T saturates (Half stops counting at 2048), which used to make every phase mean or edge average of a
/// long Half series divide by 2048 instead of the true count.
/// </summary>
public class TimeSeriesEngineOpsCountTests
{
    [Fact]
    public void PhaseSums_Half_CountsBeyondHalfIntegerRange()
    {
        // 6000 points over 2 phases: 3000 per phase, more than a Half sum of ones can reach.
        var (sums, counts) = TimeSeriesEngineOps<Half>.PhaseSums(new Vector<Half>(6000), 2);

        Assert.Equal(2, counts.Length);
        Assert.Equal(3000.0, (double)counts[0]);
        Assert.Equal(3000.0, (double)counts[1]);
        Assert.Equal(0.0, (double)sums[0]);
    }

    [Fact]
    public void PhaseSums_UnevenPhases_CountsMatchReference()
    {
        var (_, counts) = TimeSeriesEngineOps<double>.PhaseSums(new Vector<double>(11), 4);

        Assert.Equal(new[] { 3.0, 3.0, 3.0, 2.0 }, counts.ToArray());
    }

    [Fact]
    public void InRangeCounts_MatchesReference()
    {
        var counts = TimeSeriesEngineOps<double>.InRangeCounts(7, 2);

        // [i - 2, i + 2] clipped to [0, 7)
        Assert.Equal(new[] { 3.0, 4.0, 5.0, 5.0, 5.0, 4.0, 3.0 }, counts.ToArray());
    }

    [Fact]
    public void PopulationMeanStd_Half_AccumulatesInDouble()
    {
        // Autoformer and Informer standardize with these. 5000 values alternating 100 and 102: mean 101,
        // population std 1 (master, in double). Their Half sum, 505000, is far past Half's 65504.
        var series = new Vector<Half>(5000);
        for (int i = 0; i < series.Length; i++)
        {
            series[i] = (Half)(i % 2 == 0 ? 100.0f : 102.0f);
        }

        var (mean, std) = TimeSeriesEngineOps<Half>.PopulationMeanStd(series);

        Assert.Equal(101.0, mean, 10);
        Assert.Equal(1.0, std, 10);
    }

    [Fact]
    public void StlMovingAverage_Half_EdgeAverageDividesByTrueCount()
    {
        // Window 4999 over 5000 points: position 0 averages [0, 2499], i.e. 2500 points, one of them 1.
        var data = new Vector<Half>(5000);
        data[0] = (Half)1.0f;
        var model = new STLDecomposition<Half>(new STLDecompositionOptions<Half> { SeasonalPeriod = 2 });
        var method = typeof(STLDecomposition<Half>).GetMethod("MovingAverage", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);
        var result = method?.Invoke(model, new object[] { data, 4999 }) as Vector<Half>;
        Assert.NotNull(result);

        double expected = (double)(Half)(1.0 / 2500.0);
        Assert.Equal(expected, (double)(result ?? new Vector<Half>(1))[0], 6);
    }
}

public class TimeSeriesHalfPrecisionTests
{
    private sealed class MetricsProbe : ARModel<Half>
    {
        public MetricsProbe() : base(new ARModelOptions<Half> { AROrder = 1 })
        {
        }

        public Dictionary<string, Half> Metrics(Vector<Half> predictions, Vector<Half> actuals)
            => CalculateErrorMetrics(predictions, actuals);
    }

    [Fact]
    public void CalculateErrorMetrics_Half_ZeroActualSkipsMape()
    {
        // An actual of exactly zero makes the percentage error undefined, so MAPE must be omitted.
        // The 1e-10 guard rounds to zero in Half; compared in T it let the zero through and MAPE was +inf.
        var predictions = new Vector<Half>(new[] { (Half)1, (Half)2 });
        var actuals = new Vector<Half>(new[] { (Half)0, (Half)2 });

        var metrics = new MetricsProbe().Metrics(predictions, actuals);

        Assert.False(metrics.ContainsKey("MAPE"));
        Assert.Equal(0.5, (double)metrics["MAE"], 6);
    }

    [Fact]
    public void TiDE_Half_TargetStatisticsAccumulateBeyondHalfRange()
    {
        // 300 targets of 1000 sum to 300000, past Half's 65504: a Half sum made the target mean +inf,
        // every prediction non-finite and the output guard returned 0. In double the mean is 1000.
        const int rows = 300;
        var x = new Matrix<Half>(rows, 4);
        var y = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            y[i] = (Half)1000;
            for (int j = 0; j < 4; j++)
            {
                x[i, j] = (Half)1000;
            }
        }

        var model = new TiDEModel<Half>(new TiDEOptions<Half>
        {
            LookbackWindow = 4,
            HiddenSize = 4,
            Epochs = 1,
            BatchSize = 32,
        });
        model.Train(x, y);

        double prediction = (double)model.PredictSingle(new Vector<Half>(new[] { (Half)1000, (Half)1000, (Half)1000, (Half)1000 }));

        Assert.InRange(prediction, 900.0, 1100.0);
    }

    [Fact]
    public void NLinear_Half_ScalerStatisticsAccumulateBeyondHalfRange()
    {
        // 300 rows of 1000 sum past Half's 65504: a Half sum made the scaler mean +inf and the
        // non-finite forecast was replaced by 0. In double the mean is 1000 and the constant
        // series forecasts itself.
        const int rows = 300;
        var x = new Matrix<Half>(rows, 4);
        var y = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            y[i] = (Half)1000;
            for (int j = 0; j < 4; j++)
            {
                x[i, j] = (Half)1000;
            }
        }

        var model = new NLinearModel<Half>(new NLinearOptions<Half> { LookbackWindow = 4, Epochs = 1, BatchSize = 32 });
        model.Train(x, y);

        double prediction = (double)model.PredictSingle(new Vector<Half>(new[] { (Half)1000, (Half)1000, (Half)1000, (Half)1000 }));

        Assert.InRange(prediction, 990.0, 1010.0);
    }
}
