using System;
using System.Collections.Generic;
using AiDotNet.SpeechRecognition;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// Checks the engine-op greedy CTC decoder against the frame-by-frame scalar scan the speech models
/// used before it: a "value &gt; best" scan from negative infinity, repeat collapse, blank (0) removal,
/// a token cap, and the mean softmax probability of the emitted frames as the confidence.
/// </summary>
public class CtcGreedyDecodingTests
{
    private static (List<int> tokens, double confidence) Reference(
        Tensor<double> logits, int maxTokens, int framesAfterEmission = 1)
    {
        var tokens = new List<int>();
        double totalConf = 0;
        int confCount = 0;
        int prevToken = -1;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[logits.Rank - 1] : logits.Shape[0];
        int t = 0;
        while (t < numFrames && tokens.Count < maxTokens)
        {
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? logits[t, v] : logits[v];
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }

            double sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? logits[t, v] : logits[v];
                sumExp += Math.Exp(val - maxVal);
            }

            double frameConf = 1.0 / sumExp;
            if (maxIdx != prevToken && maxIdx > 0)
            {
                tokens.Add(maxIdx);
                totalConf += frameConf;
                confCount++;
                t += framesAfterEmission;
            }
            else
            {
                t++;
            }

            prevToken = maxIdx;
        }

        return (tokens, confCount > 0 ? totalConf / confCount : 0.0);
    }

    /// <summary>Small integer logits, so equal maxima within a frame are common.</summary>
    private static Tensor<double> TiedLogits(int frames, int vocab, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[frames * vocab];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 4);
        return new Tensor<double>(data, new[] { frames, vocab });
    }

    private static void AssertSame((List<int> tokens, double confidence) expected, (List<int> tokens, double confidence) actual)
    {
        Assert.Equal(expected.tokens, actual.tokens);
        if (double.IsNaN(expected.confidence))
            Assert.True(double.IsNaN(actual.confidence), $"Expected NaN confidence, got {actual.confidence}.");
        else
            Assert.Equal(expected.confidence, actual.confidence, 12);
    }

    [Theory]
    [InlineData(1, 5, int.MaxValue)]
    [InlineData(40, 6, int.MaxValue)]
    [InlineData(40, 6, 3)]
    [InlineData(25, 1, int.MaxValue)]
    [InlineData(25, 4, 0)]
    public void MatchesTheScalarScan_IncludingTies(int frames, int vocab, int maxTokens)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var logits = TiedLogits(frames, vocab, seed);
            AssertSame(Reference(logits, maxTokens), CtcGreedyDecoding.Decode(logits, maxTokens));
        }
    }

    [Fact]
    public void LowestIndexWinsATie()
    {
        var logits = new Tensor<double>(new double[] { 0, 2, 2, 1 }, new[] { 1, 4 });
        var (tokens, _) = CtcGreedyDecoding.Decode(logits, int.MaxValue);
        Assert.Equal(new List<int> { 1 }, tokens);
    }

    [Fact]
    public void NaNAndNegativeInfinityNeverWin()
    {
        double nan = double.NaN;
        double negInf = double.NegativeInfinity;
        var logits = new Tensor<double>(
            new[]
            {
                nan, 1.0, 3.0, nan,       // NaN skipped for the argmax, but poisons that frame's confidence
                negInf, negInf, negInf, 2.0,
                nan, nan, nan, nan,       // nothing can win: the blank
                negInf, negInf, negInf, negInf,
                0.5, 0.5, double.PositiveInfinity, 0.5,
            },
            new[] { 5, 4 });

        AssertSame(Reference(logits, int.MaxValue), CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void ASingleFrameVectorDecodesAsOneFrame()
    {
        var logits = new Tensor<double>(new[] { 0.1, 0.2, 0.9, 0.3 }, new[] { 4 });
        AssertSame(Reference(logits, int.MaxValue), CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void RankThreeLogitsAreRejectedLikeTheTwoIndexRead()
    {
        var logits = new Tensor<double>(new[] { 2, 3, 4 });
        Assert.Throws<ArgumentException>(() => CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void FlatDecodingReadsTheLeadingFramesTimesVocabElements()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var data = new double[2 * 3 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 3);
        var rankThree = new Tensor<double>(data, new[] { 2, 3, 4 });

        // frames = 2, vocab = 4: the first 8 flat elements, as two rows.
        var leadingData = new double[8];
        Array.Copy(data, leadingData, 8);
        var leading = new Tensor<double>(leadingData, new[] { 2, 4 });
        AssertSame(Reference(leading, int.MaxValue), CtcGreedyDecoding.DecodeFlat(rankThree, int.MaxValue));
    }

    [Theory]
    [InlineData(2, int.MaxValue)]
    [InlineData(1, 5)]
    [InlineData(0, int.MaxValue)]
    public void FrameSkippingMatchesTheScalarWalk(int framesAfterEmission, int maxTokens)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var logits = TiedLogits(30, 5, seed + 20);
            AssertSame(
                Reference(logits, maxTokens, framesAfterEmission),
                CtcGreedyDecoding.DecodeSkippingAfterEmission(logits, maxTokens, framesAfterEmission));
        }
    }
}
