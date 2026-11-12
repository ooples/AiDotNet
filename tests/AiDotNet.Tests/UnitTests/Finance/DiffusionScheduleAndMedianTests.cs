using System;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Pins the diffusion foundation models' noise schedule and sample median to the reference
/// scalar algorithms: a linear beta schedule with a running alpha-bar product, an index error for an
/// empty schedule, and a comparison-sort median whose NaN placement is the reference sort's.
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
}
