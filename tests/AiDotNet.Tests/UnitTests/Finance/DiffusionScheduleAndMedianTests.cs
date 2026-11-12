using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Finance.Evaluation;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Tests.IntegrationTests.Finance;
using Moq;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Pins the diffusion foundation models' noise schedule and sample median to the reference
/// scalar algorithms: a linear beta schedule with a running alpha-bar product, an index error for an
/// empty schedule, and a comparison-sort median whose NaN placement is the reference sort's. Also pins
/// GIFT-Eval's CRPS to the gluonts mean weighted quantile loss and PatchTST's RevIN statistics to the
/// RevIN reference (Kim et al. 2022).
/// </summary>
public class DiffusionScheduleAndMedianTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

    private static NeuralNetworkArchitecture<double> Arch(int input, int output) =>
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: input,
            outputSize: output);

    /// <summary>The reference median: copy each column, sort it with the NaN-tolerant comparer, take the middle.</summary>
    private static double[] ReferenceMedian(double[] paths, int samples, int outputLen)
    {
        var result = new double[outputLen];
        var column = new double[samples];
        for (int i = 0; i < outputLen; i++)
        {
            for (int s = 0; s < samples; s++) column[s] = paths[s * outputLen + i];
            Array.Sort(column, (a, b) => a < b ? -1 : b < a ? 1 : 0);
            result[i] = (samples % 2) == 1
                ? column[samples / 2]
                : (column[samples / 2 - 1] + column[samples / 2]) / 2.0;
        }

        return result;
    }

    private static double[] Median(object model, double[] paths, int samples, int outputLen)
    {
        var method = model.GetType().GetMethod("MedianAcrossSamples", Private);
        Assert.NotNull(method);
        var tensor = new Tensor<double>(new[] { paths.Length }, new Vector<double>(paths));
        var output = method?.Invoke(model, new object[] { tensor, samples, outputLen }) as Tensor<double>;
        Assert.NotNull(output);
        return output?.ToArray() ?? Array.Empty<double>();
    }

    private static double[] Paths(int samples, int outputLen, int seed, double nanFraction, bool withTies)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var paths = new double[samples * outputLen];
        for (int k = 0; k < paths.Length; k++)
        {
            double value = withTies ? Math.Round(rng.NextDouble() * 4.0) : rng.NextDouble() * 2.0 - 1.0;
            paths[k] = rng.NextDouble() < nanFraction ? double.NaN : value;
        }

        return paths;
    }

    private static void AssertSameBits(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        }
    }

    public static TheoryData<int, double, bool> MedianCases() => new TheoryData<int, double, bool>
    {
        // Odd and even sample counts, below and above the insertion-sort cutoff (16), with and without NaN.
        { 5, 0.0, false }, { 6, 0.0, true }, { 33, 0.0, true },
        { 5, 0.3, false }, { 6, 0.3, false }, { 17, 0.2, true }, { 40, 0.25, false }, { 41, 0.5, true },
    };

    [Theory]
    [MemberData(nameof(MedianCases))]
    public void Csdi_median_matches_reference_sort_including_nan_placement(int samples, double nanFraction, bool withTies)
    {
        var model = new CSDI<double>(Arch(100, 100));
        const int outputLen = 7;
        var paths = Paths(samples, outputLen, 11 + samples, nanFraction, withTies);

        AssertSameBits(ReferenceMedian(paths, samples, outputLen), Median(model, paths, samples, outputLen));
    }

    [Theory]
    [MemberData(nameof(MedianCases))]
    public void Tsdiff_median_matches_reference_sort_including_nan_placement(int samples, double nanFraction, bool withTies)
    {
        var model = new TSDiff<double>(Arch(192, 24));
        const int outputLen = 7;
        var paths = Paths(samples, outputLen, 29 + samples, nanFraction, withTies);

        AssertSameBits(ReferenceMedian(paths, samples, outputLen), Median(model, paths, samples, outputLen));
    }

    [Fact]
    public void Median_with_a_single_nan_column_matches_reference()
    {
        var model = new CSDI<double>(Arch(100, 100));
        const int samples = 9, outputLen = 3;
        var paths = Paths(samples, outputLen, 5, 0.0, false);
        paths[4 * outputLen + 1] = double.NaN;

        AssertSameBits(ReferenceMedian(paths, samples, outputLen), Median(model, paths, samples, outputLen));
    }

    private static void AssertLinearSchedule(object model, int steps, double betaStart, double betaEnd)
    {
        var alphaBar = model.GetType().GetField("_alphasCumprod", Private)?.GetValue(model) as Vector<double>;
        var oneMinus = model.GetType().GetField("_sqrtOneMinusAlphasCumprod", Private)?.GetValue(model) as Vector<double>;
        Assert.NotNull(alphaBar);
        Assert.NotNull(oneMinus);
        if (alphaBar is null || oneMinus is null) return;

        Assert.Equal(steps, alphaBar.Length);
        double running = 1.0;
        double denom = Math.Max(1, steps - 1);
        for (int t = 0; t < steps; t++)
        {
            double beta = betaStart + (betaEnd - betaStart) * t / denom;
            running *= 1.0 - beta;
            Assert.Equal(running, alphaBar[t], 12);
            Assert.Equal(Math.Sqrt(1.0 - running), oneMinus[t], 12);
        }
    }

    [Fact]
    public void Mgtsd_linear_schedule_matches_reference()
    {
        var options = new AiDotNet.Models.Options.MGTSDOptions<double>();
        var model = new MGTSD<double>(Arch(168, 24), options);
        AssertLinearSchedule(model, options.DiffusionSteps, options.BetaStart, options.BetaEnd);
    }

    [Fact]
    public void Tsdiff_linear_schedule_matches_reference()
    {
        var options = new AiDotNet.Models.Options.TSDiffOptions<double>();
        var model = new TSDiff<double>(Arch(192, 24), options);
        AssertLinearSchedule(model, options.NumDiffusionSteps, options.BetaStart, options.BetaEnd);
    }

    private static void AssertEmptyScheduleThrows(object model, string stepsField)
    {
        var field = model.GetType().GetField(stepsField, Private);
        var method = model.GetType().GetMethod("ComputeNoiseSchedule", Private);
        Assert.NotNull(field);
        Assert.NotNull(method);
        field?.SetValue(model, 0);

        var thrown = Assert.Throws<TargetInvocationException>(() => method?.Invoke(model, Array.Empty<object>()));
        var inner = Assert.IsType<ArgumentOutOfRangeException>(thrown.InnerException);
        Assert.Equal("index", inner.ParamName);
    }

    [Fact]
    public void Mgtsd_empty_schedule_raises_the_index_error()
        => AssertEmptyScheduleThrows(new MGTSD<double>(Arch(168, 24)), "_diffusionSteps");

    [Fact]
    public void Tsdiff_empty_schedule_raises_the_index_error()
        => AssertEmptyScheduleThrows(new TSDiff<double>(Arch(192, 24)), "_numDiffusionSteps");

    /// <summary>
    /// gluonts' quantile loss as GIFT-Eval uses it: 2 * |(yhat - y) * (1{y &lt;= yhat} - q)| summed over every
    /// series and step, divided by the summed |y|, then averaged over the levels.
    /// </summary>
    private static double ReferenceMeanWeightedQuantileLoss(
        IReadOnlyList<double[]> targets, IReadOnlyList<double[,]> forecasts, double[] levels)
    {
        double absoluteTarget = targets.Sum(series => series.Sum(Math.Abs));
        double total = 0;
        for (int q = 0; q < levels.Length; q++)
        {
            double loss = 0;
            for (int s = 0; s < targets.Count; s++)
            {
                for (int t = 0; t < targets[s].Length; t++)
                {
                    double y = targets[s][t];
                    double yhat = forecasts[s][t, q];
                    loss += 2 * Math.Abs((yhat - y) * ((y <= yhat ? 1.0 : 0.0) - levels[q]));
                }
            }

            total += loss / absoluteTarget;
        }

        return total / levels.Length;
    }

    private static readonly double[] Levels = { 0.1, 0.5, 0.9 };

    private static readonly double[,] FullGrid =
    {
        { 4.0, 5.5, 7.0 },
        { 3.0, 6.0, 6.5 },
        { 5.0, 5.2, 9.0 },
        { 1.0, 2.0, 2.5 },
    };

    [Fact]
    public void Gift_eval_crps_matches_the_gluonts_weighted_quantile_loss()
    {
        var actual = new[] { 5.0, -2.0, 7.5, 3.0 };
        var dict = new Dictionary<double, Tensor<double>>();
        for (int q = 0; q < Levels.Length; q++)
        {
            int level = q;
            var column = Enumerable.Range(0, actual.Length).Select(t => FullGrid[t, level]).ToArray();
            dict[Levels[q]] = new Tensor<double>(new[] { actual.Length }, new Vector<double>(column));
        }

        double crps = new GIFTEvalBenchmark<double>().ComputeCRPS(
            dict, new Tensor<double>(new[] { actual.Length }, new Vector<double>(actual)));

        Assert.Equal(ReferenceMeanWeightedQuantileLoss(new[] { actual }, new[] { FullGrid }, Levels), crps, 12);
    }

    [Fact]
    public void Gift_eval_benchmark_skips_incomplete_quantile_grids_and_aggregates_over_the_dataset()
    {
        // Series 0 (context 6) gets the full [4, 3] grid; series 1 (context 8) gets a grid that stops short,
        // which GIFT-Eval cannot score, so it must not contribute zero-filled quantiles.
        var first = new[] { 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 5.0, -2.0, 7.5, 3.0 };
        var second = new[] { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0 };
        var flatFull = new double[12];
        for (int t = 0; t < 4; t++)
        {
            for (int q = 0; q < 3; q++)
                flatFull[t * 3 + q] = FullGrid[t, q];
        }

        var model = new Mock<ITimeSeriesFoundationModel<double>>();
        model.Setup(m => m.Forecast(It.IsAny<Tensor<double>>(), It.IsAny<double[]?>()))
            .Returns((Tensor<double> context, double[]? quantiles) =>
                quantiles is null
                    ? new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 }))
                    : context.Length == 6
                        ? new Tensor<double>(new[] { 12 }, new Vector<double>(flatFull))
                        : new Tensor<double>(new[] { 5 }, new Vector<double>(new[] { 9.0, 9.0, 9.0, 9.0, 9.0 })));

        var results = new GIFTEvalBenchmark<double>().RunBenchmark(
            model.Object,
            new[]
            {
                new Tensor<double>(new[] { first.Length }, new Vector<double>(first)),
                new Tensor<double>(new[] { second.Length }, new Vector<double>(second)),
            },
            new[] { 6, 8 },
            new[] { 1, 1 },
            Levels);

        var expected = ReferenceMeanWeightedQuantileLoss(
            new[] { new[] { 5.0, -2.0, 7.5, 3.0 } }, new[] { FullGrid }, Levels);
        Assert.Equal(expected, results["CRPS"], 12);
        Assert.Equal(1.0, results["NumCRPSEvaluated"]);
        Assert.Equal(2.0, results["NumSeries"]);
    }

    public static TheoryData<Type> RevinModels() => new TheoryData<Type>
    {
        typeof(PatchTST<double>),
        typeof(FEDformer<double>),
        typeof(ITransformer<double>),
    };

    private static object RevinModel(Type modelType, int features) =>
        FinanceModelTestFactory.CreateNativeModel<double>(
            modelType, options => options.GetType().GetProperty("NumFeatures")?.SetValue(options, features));

    private static double[] InvokeStat(object model, string name, params object[] args)
    {
        var method = model.GetType().GetMethod(name, Private);
        Assert.NotNull(method);
        var output = method?.Invoke(model, args) as Tensor<double>;
        Assert.NotNull(output);
        return output?.ToArray() ?? Array.Empty<double>();
    }

    [Theory]
    [MemberData(nameof(RevinModels))]
    public void Revin_statistics_use_each_features_own_sample_count(Type modelType)
    {
        // Element i belongs to feature i % 3; the trailing partial row reaches features 0 and 1 only, so they
        // have three values each and feature 2 has two. RevIN: mean over time, stdev = sqrt(var + 1e-5).
        const int features = 3;
        var values = new[] { 2.0, 4.0, 6.0, 8.0, -1.0, 3.0, 5.0, 10.0 };
        var model = RevinModel(modelType, features);
        var input = new Tensor<double>(new[] { values.Length }, new Vector<double>(values));

        var mean = InvokeStat(model, "CalculateInstanceMean", input);
        var meanTensor = new Tensor<double>(new[] { features }, new Vector<double>(mean));
        var stdev = InvokeStat(model, "CalculateInstanceStd", input, meanTensor);

        for (int f = 0; f < features; f++)
        {
            int feature = f;
            var own = values.Where((_, i) => i % features == feature).ToArray();
            double expectedMean = own.Average();
            double variance = own.Select(v => (v - expectedMean) * (v - expectedMean)).Sum() / own.Length;
            Assert.Equal(expectedMean, mean[f], 12);
            Assert.Equal(Math.Sqrt(variance + 1e-5), stdev[f], 12);
        }

        var normalized = InvokeStat(model, "ApplyRevIN", input, true);
        for (int i = 0; i < values.Length; i++)
            Assert.Equal((values[i] - mean[i % features]) / stdev[i % features], normalized[i], 12);
    }

    [Theory]
    [MemberData(nameof(RevinModels))]
    public void Revin_rejects_an_input_without_a_value_for_every_feature(Type modelType)
    {
        var model = RevinModel(modelType, 3);
        var input = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 1.0, 2.0 }));
        var method = model.GetType().GetMethod("CalculateInstanceMean", Private);
        Assert.NotNull(method);
        var thrown = Assert.Throws<TargetInvocationException>(() => method?.Invoke(model, new object[] { input }));
        Assert.IsType<ArgumentException>(thrown.InnerException);
    }
}
