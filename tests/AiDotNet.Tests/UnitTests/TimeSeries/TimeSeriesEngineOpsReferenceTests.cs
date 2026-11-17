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

    [Fact]
    public void DLinear_Half_TrainsInDoubleWithoutOverflow()
    {
        // Windows of 500 with targets of 600: one batch gradient is 32 * 500 * 100 = 1.6e6, past
        // Half's 65504, and the step 1e-7 / 32 rounds to 0 in Half, so training in T made every weight
        // NaN and the forecast fell back to 0. In double the forecast moves from 500 toward 600.
        const int rows = 300;
        var x = new Matrix<Half>(rows, 4);
        var y = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            y[i] = (Half)600;
            for (int j = 0; j < 4; j++)
            {
                x[i, j] = (Half)500;
            }
        }

        var model = new DLinearModel<Half>(new DLinearOptions<Half>
        {
            LookbackWindow = 4,
            MovingAverageKernel = 3,
            LearningRate = 1e-7,
            Epochs = 1,
        });
        model.Train(x, y);

        double prediction = (double)model.PredictSingle(new Vector<Half>(new[] { (Half)500, (Half)500, (Half)500, (Half)500 }));

        Assert.InRange(prediction, 501.0, 601.0);
    }

    [Fact]
    public void GatedResidualNetwork_Half_InitialWeightsAreFinite()
    {
        // Box-Muller draws u1 = 1 - U. Drawn in Half, any U within 2^-12 of 1 rounds to 1, u1 is 0 and
        // log(0) made the weight infinite: about 4 of the ~17000 weights here. Drawn in double, all are finite.
        for (int seed = 1; seed <= 3; seed++)
        {
            var grn = new AiDotNet.TimeSeries.TFT.GatedResidualNetwork<Half>(64, 64, 64, seed);
            var parameters = grn.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                // Half.IsFinite does not exist on net471, which this project also targets.
                Assert.True(!float.IsNaN((float)parameters[i]) && !float.IsInfinity((float)parameters[i]), $"seed {seed}, parameter {i} is {parameters[i]}");
            }
        }
    }

    [Fact]
    public void NLinear_Half_SubnormalScalerPreservedAcrossPredictSerializeAndClone()
    {
        // Targets repeat [0, 0, 0, Half.Epsilon], whose population std is sqrt(3)/4 * 2^-24 ≈ 2.5809568e-8.
        // Stored in Half, that std rounds to 0; a fallback divisor of 1.0 inflated the O(1) normalized bias
        // into an O(1) prediction (~2.08) instead of Half.Epsilon (~5.96e-8).
        const int blocks = 32;
        const int rows = blocks * 4;
        Half epsHalf = Half.Epsilon;
        double epsDouble = (double)epsHalf;

        var xHalf = new Matrix<Half>(rows, 4);
        var yHalf = new Vector<Half>(rows);
        var xFloat = new Matrix<float>(rows, 4);
        var yFloat = new Vector<float>(rows);
        var xDouble = new Matrix<double>(rows, 4);
        var yDouble = new Vector<double>(rows);

        for (int b = 0; b < blocks; b++)
        {
            for (int k = 0; k < 3; k++)
            {
                int r = b * 4 + k;
                xHalf[r, 3] = epsHalf;
                xFloat[r, 3] = (float)epsDouble;
                xDouble[r, 3] = epsDouble;
                yHalf[r] = (Half)0;
                yFloat[r] = 0.0f;
                yDouble[r] = 0.0;
            }

            int r3 = b * 4 + 3;
            yHalf[r3] = epsHalf;
            yFloat[r3] = (float)epsDouble;
            yDouble[r3] = epsDouble;
        }

        var halfModel = new NLinearModel<Half>(new NLinearOptions<Half>
        {
            LookbackWindow = 4,
            Epochs = 80,
            BatchSize = 16,
            LearningRate = 0.05,
            Seed = 42
        });
        var floatModel = new NLinearModel<float>(new NLinearOptions<float>
        {
            LookbackWindow = 4,
            Epochs = 80,
            BatchSize = 16,
            LearningRate = 0.05,
            Seed = 42
        });
        var doubleModel = new NLinearModel<double>(new NLinearOptions<double>
        {
            LookbackWindow = 4,
            Epochs = 80,
            BatchSize = 16,
            LearningRate = 0.05,
            Seed = 42
        });

        halfModel.Train(xHalf, yHalf);
        floatModel.Train(xFloat, yFloat);
        doubleModel.Train(xDouble, yDouble);

        var zeroWindowHalf = new Vector<Half>(new[] { (Half)0, (Half)0, (Half)0, (Half)0 });
        var pulseWindowHalf = new Vector<Half>(new[] { (Half)0, (Half)0, (Half)0, epsHalf });
        var zeroWindowFloat = new Vector<float>(new[] { 0.0f, 0.0f, 0.0f, 0.0f });
        var pulseWindowFloat = new Vector<float>(new[] { 0.0f, 0.0f, 0.0f, (float)epsDouble });
        var zeroWindowDouble = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var pulseWindowDouble = new Vector<double>(new[] { 0.0, 0.0, 0.0, epsDouble });

        double refZero = doubleModel.PredictSingle(zeroWindowDouble);
        double refPulse = doubleModel.PredictSingle(pulseWindowDouble);
        Assert.InRange(refZero, 0.8 * epsDouble, 1.2 * epsDouble);
        Assert.InRange(Math.Abs(refPulse), 0.0, 0.2 * epsDouble);

        // 1. Half training + PredictSingle preserves subnormal std = 2.58e-8 (returns Half.Epsilon, not ~2.08 or 0).
        Assert.Equal(epsHalf, halfModel.PredictSingle(zeroWindowHalf));
        Assert.Equal((Half)0, halfModel.PredictSingle(pulseWindowHalf));

        // 2. Public Half persistence must preserve both parameters and exact double scaler state.
        // Populate metrics so the shared serializer also exercises its numeric metric values.
        Assert.NotEmpty(halfModel.EvaluateModel(xHalf, yHalf));
        var restoredHalf = new NLinearModel<Half>(new NLinearOptions<Half>
        {
            LookbackWindow = 4,
            Epochs = 80,
            BatchSize = 16,
            LearningRate = 0.05,
            Seed = 42
        });
        restoredHalf.Deserialize(halfModel.Serialize());
        var clonedHalf = (NLinearModel<Half>)halfModel.Clone();
        var deepCopiedHalf = (NLinearModel<Half>)halfModel.DeepCopy();
        foreach (var candidate in new[] { restoredHalf, clonedHalf, deepCopiedHalf })
        {
            Assert.Equal(epsHalf, candidate.PredictSingle(zeroWindowHalf));
            Assert.Equal((Half)0, candidate.PredictSingle(pulseWindowHalf));
        }

        // 3. Full Serialize/Deserialize, Clone, and DeepCopy roundtrip at the same std = 2.58e-8 (< old 1e-8 cutoff).
        var reloadedFloat = new NLinearModel<float>(new NLinearOptions<float>
        {
            LookbackWindow = 4,
            Epochs = 80,
            BatchSize = 16,
            LearningRate = 0.05,
            Seed = 42
        });
        reloadedFloat.Deserialize(floatModel.Serialize());

        var clonedFloat = (NLinearModel<float>)floatModel.Clone();
        var deepCopiedFloat = (NLinearModel<float>)floatModel.DeepCopy();

        foreach (var candidate in new[] { floatModel, reloadedFloat, clonedFloat, deepCopiedFloat })
        {
            double predZero = candidate.PredictSingle(zeroWindowFloat);
            double predPulse = candidate.PredictSingle(pulseWindowFloat);

            Assert.InRange(predZero, 0.8 * epsDouble, 1.2 * epsDouble);
            Assert.InRange(Math.Abs(predPulse), 0.0, 0.2 * epsDouble);
        }
    }

    [Fact]
    public void GARCH_Half_EstimatesNondegenerateParametersMatchingDouble()
    {
        // Simulated GARCH(1,1) volatility-clustering shocks (omega = 0.05, alpha = 0.35, beta = 0.50),
        // quantized to Half so Half and double models observe identical input values.
        // Previously in Half, omega ≈ 0.01 had ULP ≈ 7.63e-6 > epsilon = 1e-6, making plusOmega - minusOmega == 0,
        // poisoning omega with 0/0 = NaN, and hiding the failed fit by clamping NaN to VarianceFloor (1e-5).
        const int n = 180;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(12345);
        var xHalf = new Matrix<Half>(n, 1);
        var yHalf = new Vector<Half>(n);
        var xDouble = new Matrix<double>(n, 1);
        var yDouble = new Vector<double>(n);

        double sigma2 = 0.05 / (1.0 - 0.35 - 0.50);
        for (int t = 0; t < n; t++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double shock = Math.Sqrt(sigma2) * z;
            Half h = (Half)shock;
            double q = (double)h;

            xHalf[t, 0] = (Half)(t % 16);
            yHalf[t] = h;
            xDouble[t, 0] = (double)(Half)(t % 16);
            yDouble[t] = q;

            sigma2 = 0.05 + 0.35 * (q * q) + 0.50 * sigma2;
        }

        var halfModel = new GARCHModel<Half>(new GARCHModelOptions<Half>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MeanModel = new ARIMAModel<Half>(new ARIMAOptions<Half> { P = 0, D = 0, Q = 0 })
        });
        var doubleModel = new GARCHModel<double>(new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MeanModel = new ARIMAModel<double>(new ARIMAOptions<double> { P = 0, D = 0, Q = 0 })
        });

        halfModel.Train(xHalf, yHalf);
        doubleModel.Train(xDouble, yDouble);

        var halfInfo = halfModel.GetModelMetadata().AdditionalInfo;
        var doubleInfo = doubleModel.GetModelMetadata().AdditionalInfo;

        double halfOmega = (double)((Vector<Half>)halfInfo["Omega"])[0];
        double halfAlpha = (double)((Vector<Half>)halfInfo["Alpha"])[0];
        double halfBeta = (double)((Vector<Half>)halfInfo["Beta"])[0];

        double doubleOmega = ((Vector<double>)doubleInfo["Omega"])[0];
        double doubleAlpha = ((Vector<double>)doubleInfo["Alpha"])[0];
        double doubleBeta = ((Vector<double>)doubleInfo["Beta"])[0];

        Assert.True(!double.IsNaN(halfOmega) && !double.IsInfinity(halfOmega), $"Half omega is non-finite: {halfOmega}");
        Assert.True(!double.IsNaN(halfAlpha) && !double.IsInfinity(halfAlpha), $"Half alpha is non-finite: {halfAlpha}");
        Assert.True(!double.IsNaN(halfBeta) && !double.IsInfinity(halfBeta), $"Half beta is non-finite: {halfBeta}");

        // Must not collapse to the 1e-5 variance floor (the masked NaN signature) and must move off initial [0.01, 0.05, 0.85].
        Assert.True(halfOmega > 1e-3, $"Expected non-degenerate Half omega > 1e-3, got {halfOmega}");
        Assert.True(Math.Abs(halfOmega - (double)(Half)0.01) > 5e-3, $"Half omega stayed at initial 0.01: {halfOmega}");
        Assert.True(halfAlpha > 0.0 && halfBeta >= 0.0 && halfAlpha + halfBeta < 1.0,
            $"Half GARCH parameters violated stationarity: alpha={halfAlpha}, beta={halfBeta}");

        // Half MLE parameters must closely match double MLE parameters within Half rounding tolerance.
        Assert.InRange(Math.Abs(halfOmega - doubleOmega), 0.0, 5e-3);
        Assert.InRange(Math.Abs(halfAlpha - doubleAlpha), 0.0, 1e-2);
        Assert.InRange(Math.Abs(halfBeta - doubleBeta), 0.0, 1e-2);

        // Fitted Half parameters must strictly improve the Gaussian log-likelihood over initial [0.01, 0.05, 0.85].
        double initialLl = GaussianLogLikelihoodReference(yDouble, 0.01, 0.05, 0.85);
        double fittedHalfLl = GaussianLogLikelihoodReference(yDouble, halfOmega, halfAlpha, halfBeta);
        Assert.True(fittedHalfLl > initialLl + 1.0,
            $"Expected fitted Half log-likelihood ({fittedHalfLl}) > initial ({initialLl})");

        // Anti-masking check: ClampMinimum must preserve NaN rather than turning NaN into VarianceFloor.
        var clampMethod = typeof(GARCHModel<Half>).GetMethod("ClampMinimum", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(clampMethod);
        var clamped = clampMethod?.Invoke(
            halfModel,
            new object[] { new Vector<Half>(new[] { Half.NaN }), (Half)1e-5 }) as Vector<Half>;
        Assert.NotNull(clamped);
        Assert.True(float.IsNaN((float)(clamped ?? new Vector<Half>(1))[0]));
    }

    private static double GaussianLogLikelihoodReference(Vector<double> y, double omega, double alpha, double beta)
    {
        int n = y.Length;
        double mean = 0.0;
        for (int i = 0; i < n; i++)
        {
            mean += y[i];
        }
        mean /= n;

        double sampleVar = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = y[i] - mean;
            sampleVar += d * d;
        }
        sampleVar /= (n - 1);

        var sigma2 = new double[n];
        sigma2[0] = sampleVar;
        double sum = 0.0;
        for (int t = 1; t < n; t++)
        {
            double prev = y[t - 1] - mean;
            double rt = y[t] - mean;
            double v = omega + alpha * (prev * prev) + beta * sigma2[t - 1];
            sigma2[t] = v < 1e-5 ? 1e-5 : v;
            sum += Math.Log(sigma2[t]) + (rt * rt) / sigma2[t];
        }

        return -0.5 * sum;
    }
}
