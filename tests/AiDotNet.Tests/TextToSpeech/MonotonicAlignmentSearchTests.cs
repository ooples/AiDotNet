using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The engine-op monotonic alignment search of <see cref="TtsModelBase{T}"/> against a line-for-line port of the
/// Glow-TTS reference (<c>monotonic_align/core.pyx</c>, <c>maximum_path_each</c>).
/// </summary>
public sealed class MonotonicAlignmentSearchTests
{
    // Glow-TTS maximum_path_each: value accumulated in place, then the path traced back from the last token.
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

    private static Tensor<T> ToTensor<T>(double[,] scores, Func<double, T> convert)
    {
        int tokens = scores.GetLength(0), frames = scores.GetLength(1);
        var flat = new T[tokens * frames];
        for (int i = 0; i < tokens; i++)
            for (int j = 0; j < frames; j++) flat[i * frames + j] = convert(scores[i, j]);
        return new Tensor<T>(flat, new[] { tokens, frames });
    }

    [Fact]
    public void Durations_MatchTheGlowTtsReference_OnRandomScores()
    {
        var random = RandomHelper.CreateSeededRandom(2024);
        for (int trial = 0; trial < 60; trial++)
        {
            int tokens = 1 + random.Next(6), frames = tokens + random.Next(9);
            var scores = new double[tokens, frames];
            for (int i = 0; i < tokens; i++)
                for (int j = 0; j < frames; j++) scores[i, j] = random.Next(-10, 11);   // integers: exact in float, with ties

            var expected = ReferenceDurations(scores);
            Assert.Equal(expected, TtsModelBase<double>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => v)));
            Assert.Equal(expected, TtsModelBase<float>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => (float)v)));
        }
    }

    [Fact]
    public void Durations_FollowTheBestPath_NotTheGreedyOne()
    {
        // Choosing the better next cell frame by frame stays on token 0 for the 10 and reaches the 100 too late
        // (durations [3, 1, 1], total 11); the best monotonic path is [2, 1, 2] (0 + 10 + 0 + 100 + 1 = 111).
        var scores = new double[,]
        {
            { 0, 10, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { 0, 1, 1, 100, 1 },
        };
        var expected = ReferenceDurations(scores);
        Assert.Equal(new[] { 2, 1, 2 }, expected);
        Assert.Equal(expected, TtsModelBase<double>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => v)));
    }

    [Fact]
    public void Ties_StayOnTheSameToken_AsTheReference()
    {
        var scores = new double[3, 7];
        var expected = ReferenceDurations(scores);
        Assert.Equal(expected, TtsModelBase<double>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => v)));
    }

    [Fact]
    public void NonFiniteScoresInTheBand_AreRejected()
    {
        // Two tokens over three frames: frame 1 can hold either token, so (token 0, frame 1) is in the band.
        foreach (double bad in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        {
            var scores = new double[,] { { 0, bad, 0 }, { 0, 0, 0 } };
            Assert.Throws<ArgumentException>(() => MonotonicAlignment.MaximumPath(scores));
            Assert.Throws<ArgumentException>(() => TtsModelBase<double>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => v)));
            Assert.Throws<ArgumentException>(() => TtsModelBase<float>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => (float)v)));
        }
    }

    [Fact]
    public void NonFiniteScoresOutsideTheBand_AreNotRead()
    {
        // Two tokens over two frames leave one path, the diagonal: (token 0, frame 1) and (token 1, frame 0) are
        // never read, so a non-finite score there changes nothing, exactly as in MonotonicAlignment.MaximumPath.
        foreach (double bad in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        {
            var scores = new double[,] { { 0, bad }, { bad, 0 } };
            var expected = MonotonicAlignment.MaximumPath(scores);
            Assert.Equal(new[] { 1, 1 }, expected);
            Assert.Equal(expected, TtsModelBase<double>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => v)));
            Assert.Equal(expected, TtsModelBase<float>.MonotonicDurations(AiDotNetEngine.Current, ToTensor(scores, v => (float)v)));
        }
    }
}
