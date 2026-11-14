using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// References for the shared TtsModelBase helpers in Half: each case overflowed or lost its index when the value or
/// the index was carried in T. Ported from #2538's TtsNarrowPrecisionReferenceTests.
/// </summary>
public sealed class TtsModelBaseNarrowPrecisionTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    // Glow-TTS maximum_path_each, in double.
    private static int[] ReferenceDurations(double[,] scores)
    {
        const double MaxNegative = -1e9;
        int tx = scores.GetLength(0), ty = scores.GetLength(1);
        var value = (double[,])scores.Clone();
        for (int y = 0; y < ty; y++)
            for (int x = Math.Max(0, tx + y - ty); x < Math.Min(tx, y + 1); x++)
            {
                double vCur = x == y ? MaxNegative : value[x, y - 1];
                double vPrev = x == 0 ? (y == 0 ? 0.0 : MaxNegative) : value[x - 1, y - 1];
                value[x, y] = Math.Max(vCur, vPrev) + value[x, y];
            }
        var durations = new int[tx];
        int index = tx - 1;
        for (int y = ty - 1; y >= 0; y--)
        {
            durations[index]++;
            if (index != 0 && (index == y || value[index, y - 1] < value[index - 1, y - 1])) index--;
        }
        return durations;
    }

    [Fact]
    public void MonotonicDurations_InHalf_DoNotOverflowTheAccumulatedPathScores()
    {
        // Accumulated along 40 frames the path scores reach -80,000, past Half's -65,504: without renormalization
        // both tokens' values become -inf, the trace-back compares -inf < -inf and the path is lost.
        const int frames = 40;
        var scores = new double[2, frames];
        for (int y = 0; y < frames; y++)
        {
            scores[0, y] = y < 38 ? -2000 : -4000;
            scores[1, y] = y < 38 ? -4000 : -2000;
        }
        var expected = ReferenceDurations(scores);
        Assert.Equal(new[] { 38, 2 }, expected);

        var flat = new Half[2 * frames];
        for (int i = 0; i < 2; i++)
            for (int y = 0; y < frames; y++) flat[i * frames + y] = (Half)scores[i, y];
        Assert.Equal(expected, TtsModelBase<Half>.MonotonicDurations(Engine, new Tensor<Half>(flat, new[] { 2, frames })));
    }

    [Fact]
    public void FirstArgMax_InHalf_ReturnsIndicesPastHalfRange()
    {
        // An index carried in Half is inexact past 2,048 and infinite past 65,504.
        var values = new Half[70000];
        values[60000] = (Half)1.0;
        values[65000] = (Half)1.0;
        Assert.Equal(60000, TtsModelBase<Half>.FirstArgMax(Engine, new Tensor<Half>(values, new[] { values.Length })));

        var small = new Half[] { (Half)0.5, (Half)2.0, (Half)2.0, (Half)1.0 };
        Assert.Equal(1, TtsModelBase<Half>.FirstArgMax(Engine, new Tensor<Half>(small, new[] { 4 })));
    }

    [Theory]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NaN)]
    [InlineData(-65504.0)]
    public void CheckedInt32_InHalf_FollowsCheckedConversion(double value)
    {
        // -2^31 rounds to -inf in Half, so a bound compared in Half lets -inf through; NaN must throw as well.
        var tensor = new Tensor<Half>(new[] { (Half)3.0, (Half)value }, new[] { 2 });
        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            var error = Assert.Throws<OverflowException>(() => TtsModelBase<Half>.CheckedInt32(Engine, tensor, "overflow"));
            Assert.Equal("overflow", error.Message);
        }
        else
        {
            Assert.Equal(new[] { 3, (int)value }, TtsModelBase<Half>.CheckedInt32(Engine, tensor, "overflow"));
        }
    }

    [Fact]
    public void CheckedInt32_InDouble_RejectsValuesOutsideInt32()
    {
        Assert.Throws<OverflowException>(() =>
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { 2147483648.0 }, new[] { 1 }), "m"));
        Assert.Throws<OverflowException>(() =>
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { -2147483649.0 }, new[] { 1 }), "m"));
        Assert.Equal(new[] { int.MinValue, int.MaxValue },
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { -2147483648.0, 2147483647.0 }, new[] { 2 }), "m"));
    }
}
