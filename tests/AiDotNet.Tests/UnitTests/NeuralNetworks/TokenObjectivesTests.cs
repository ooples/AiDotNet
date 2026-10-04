using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Closed-form checks of the token objectives every generative sequence model shares (NeuralNetworkBase).
/// </summary>
public class TokenObjectivesTests
{
    private static Tensor<double> Logits(int rows, int vocab, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { rows, vocab });
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 4) - 2;
        return t;
    }

    private static double LogSoftmax(Tensor<double> logits, int row, int v)
    {
        int vocab = logits.Shape[1];
        double max = Enumerable.Range(0, vocab).Max(c => logits[row, c]);
        double logSum = max + Math.Log(Enumerable.Range(0, vocab).Sum(c => Math.Exp(logits[row, c] - max)));
        return logits[row, v] - logSum;
    }

    [Fact]
    public void TokenCrossEntropy_IsTheMeanNegativeLogLikelihoodFromTheFirstRow()
    {
        var logits = Logits(5, 7, 1);
        int[] labels = { 3, 0, 6 };
        double expected = -Enumerable.Range(0, 3).Sum(t => LogSoftmax(logits, 2 + t, labels[t])) / 3;
        var loss = NeuralNetworkBase<double>.TokenCrossEntropy(logits, labels, firstRow: 2);
        Assert.Equal(expected, loss[0], 10);
    }

    [Fact]
    public void TokenCrossEntropy_RejectsLabelsPastTheLogits()
    {
        var logits = Logits(2, 4, 2);
        Assert.Throws<ArgumentException>(() => NeuralNetworkBase<double>.TokenCrossEntropy(logits, new[] { 1, 2 }, firstRow: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => NeuralNetworkBase<double>.TokenCrossEntropy(logits, new[] { 4 }));
    }

    [Fact]
    public void SoftTargetCrossEntropy_IsTheScaledNegativeExpectedLogProbability()
    {
        var logits = Logits(2, 5, 3);
        var target = new Tensor<double>(new[] { 2, 5 });
        target[0, 1] = 0.25; target[0, 4] = 0.75; target[1, 2] = 1.0;
        double expected = -0.5 * ((0.25 * LogSoftmax(logits, 0, 1)) + (0.75 * LogSoftmax(logits, 0, 4)) + LogSoftmax(logits, 1, 2));
        var loss = NeuralNetworkBase<double>.SoftTargetCrossEntropy(logits, target, scale: 0.5);
        Assert.Equal(expected, loss[0], 10);
    }

    [Fact]
    public void GreedyToken_TakesTheFirstMaximum()
    {
        var logits = new Tensor<double>(new[] { 2, 4 });
        logits[1, 1] = 3.0; logits[1, 3] = 3.0; logits[1, 2] = -1.0;
        Assert.Equal(1, NeuralNetworkBase<double>.GreedyToken(logits, 1));
    }

    [Fact]
    public void GreedyDecode_FeedsEachTokenBack_AndStopsOnTheStopToken()
    {
        // The model's next token is always (last context token + 1); token 4 stops decoding.
        var seen = new List<int>();
        Tensor<double> Next(List<int> context)
        {
            seen.Add(context.Count);
            var logits = new Tensor<double>(new[] { context.Count, 8 });
            logits[context.Count - 1, context[^1] + 1] = 1.0;
            return logits;
        }
        var context = new List<int> { 1 };
        var generated = NeuralNetworkBase<double>.GreedyDecode(Next, context, maxSteps: 10, (token, _) => token == 4);
        Assert.Equal(new[] { 2, 3, 4 }, generated);
        Assert.Equal(new[] { 1, 2, 3 }, seen);
        Assert.Equal(new[] { 1, 2, 3 }, context);
    }

    [Fact]
    public void GreedyDecode_StopsAtMaxSteps()
    {
        Tensor<double> Next(List<int> context) => new(new[] { context.Count, 3 });
        var generated = NeuralNetworkBase<double>.GreedyDecode(Next, new List<int> { 0 }, maxSteps: 2, (_, _) => false);
        Assert.Equal(2, generated.Count);
    }

    [Fact]
    public void ShiftRight_PrependsTheStartToken()
    {
        Assert.Equal(new[] { 9, 5, 6 }, NeuralNetworkBase<double>.ShiftRight(new[] { 5, 6, 7 }, 9));
        Assert.Empty(NeuralNetworkBase<double>.ShiftRight(Array.Empty<int>(), 9));
    }

    [Fact]
    public void TokenIds_RoundClampAndCap()
    {
        var target = new Tensor<double>(new[] { 4 });
        target[0] = 2.4; target[1] = 2.6; target[2] = 99; target[3] = 1;
        Assert.Equal(new[] { 2, 3, 9 }, NeuralNetworkBase<double>.TokenIds(target, id => Math.Min(id, 9), maxCount: 3));
        Assert.Equal(new[] { 2.0, 3.0 }, NeuralNetworkBase<double>.TokenTensor(new[] { 2, 3 }).ToArray());
    }
}
