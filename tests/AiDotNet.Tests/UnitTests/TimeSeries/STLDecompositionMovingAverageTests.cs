using System;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

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
