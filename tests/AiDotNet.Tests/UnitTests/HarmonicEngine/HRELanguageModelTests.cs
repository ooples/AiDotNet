using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Smoke tests for the HRE transformer-replacement architecture
/// (SequenceAxisIMDAttention → SpectralGatingFFN → HREBlock → HRELanguageModel).
/// These tests verify end-to-end forward passes produce valid outputs before we
/// wire up the no-backprop training strategies.
/// </summary>
public class HRELanguageModelTests
{
    private readonly ITestOutputHelper _output;

    public HRELanguageModelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void SequenceAxisIMDAttention_Forward_ProducesValidOutput()
    {
        const int seqLen = 8;
        const int embedDim = 16;
        const int fftSize = 1024;

        var layer = new SequenceAxisIMDAttention<double>(seqLen, embedDim, fftSize);

        var input = new Tensor<double>([seqLen, embedDim]);
        var rng = RandomHelper.CreateSecureRandom();
        for (int s = 0; s < seqLen; s++)
            for (int e = 0; e < embedDim; e++)
                input[s, e] = rng.NextDouble() * 2 - 1;

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(seqLen, output.Shape[0]);
        Assert.Equal(embedDim, output.Shape[1]);

        double maxMagnitude = 0;
        for (int s = 0; s < seqLen; s++)
        {
            for (int e = 0; e < embedDim; e++)
            {
                Assert.False(double.IsNaN(output[s, e]), $"Output[{s},{e}] is NaN");
                Assert.False(double.IsInfinity(output[s, e]), $"Output[{s},{e}] is Infinity");
                if (Math.Abs(output[s, e]) > maxMagnitude) maxMagnitude = Math.Abs(output[s, e]);
            }
        }

        _output.WriteLine($"Max output magnitude: {maxMagnitude:F4}");
        Assert.True(maxMagnitude > 1e-8, "Output should have non-trivial magnitude");

        // Verify attention weights were stored and each row sums to ~1 (softmax property)
        var attn = layer.LastAttentionWeights;
        Assert.NotNull(attn);
        for (int i = 0; i < seqLen; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < seqLen; j++) rowSum += attn[i, j];
            Assert.Equal(1.0, rowSum, 4);
        }
    }

    [Fact]
    public void SpectralGatingFFN_Forward_PreservesShape()
    {
        const int embedDim = 16;
        var ffn = new SpectralGatingFFN<double>(embedDim);

        // Test [E], [S, E], and [B, S, E] shapes
        var single = new Tensor<double>([embedDim]);
        for (int i = 0; i < embedDim; i++) single[i] = Math.Sin(i * 0.1);
        var outSingle = ffn.Forward(single);
        Assert.Equal(1, outSingle.Shape.Length);
        Assert.Equal(embedDim, outSingle.Shape[0]);

        var sequence = new Tensor<double>([8, embedDim]);
        for (int s = 0; s < 8; s++)
            for (int e = 0; e < embedDim; e++)
                sequence[s, e] = Math.Cos(s * 0.3 + e * 0.1);
        var outSeq = ffn.Forward(sequence);
        Assert.Equal(2, outSeq.Shape.Length);
        Assert.Equal(8, outSeq.Shape[0]);
        Assert.Equal(embedDim, outSeq.Shape[1]);

        var batched = new Tensor<double>([2, 8, embedDim]);
        for (int b = 0; b < 2; b++)
            for (int s = 0; s < 8; s++)
                for (int e = 0; e < embedDim; e++)
                    batched[b, s, e] = Math.Sin((b + s + e) * 0.2);
        var outBatched = ffn.Forward(batched);
        Assert.Equal(3, outBatched.Shape.Length);
        Assert.Equal(2, outBatched.Shape[0]);
        Assert.Equal(8, outBatched.Shape[1]);
        Assert.Equal(embedDim, outBatched.Shape[2]);

        // All outputs finite
        for (int i = 0; i < outBatched.Length; i++)
        {
            Assert.False(double.IsNaN(outBatched[i]));
            Assert.False(double.IsInfinity(outBatched[i]));
        }
    }

    [Fact]
    public void HREBlock_Forward_PreservesShapeAndNontrivial()
    {
        const int seqLen = 8;
        const int embedDim = 16;
        const int fftSize = 1024;

        var block = new HREBlock<double>(seqLen, embedDim, fftSize);
        _output.WriteLine($"HREBlock parameter count: {block.ParameterCount}");

        var input = new Tensor<double>([seqLen, embedDim]);
        var rng = RandomHelper.CreateSecureRandom();
        for (int s = 0; s < seqLen; s++)
            for (int e = 0; e < embedDim; e++)
                input[s, e] = rng.NextDouble() * 2 - 1;

        var output = block.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(seqLen, output.Shape[0]);
        Assert.Equal(embedDim, output.Shape[1]);

        double diff = 0;
        for (int s = 0; s < seqLen; s++)
        {
            for (int e = 0; e < embedDim; e++)
            {
                Assert.False(double.IsNaN(output[s, e]));
                Assert.False(double.IsInfinity(output[s, e]));
                diff += Math.Abs(output[s, e] - input[s, e]);
            }
        }

        // The block should transform the input (output differs from input)
        Assert.True(diff > 1e-6,
            $"HREBlock output should differ from input (total absolute diff: {diff:E4})");
    }

    [Fact]
    public void HRELanguageModel_Forward_ProducesValidLogits()
    {
        const int vocabSize = 32;
        const int seqLen = 8;
        const int embedDim = 16;
        const int numLayers = 2;
        const int fftSize = 1024;

        var model = new HRELanguageModel<double>(
            vocabSize, seqLen, embedDim, numLayers, fftSize, seed: 42);

        _output.WriteLine($"HRELanguageModel parameter count: {model.ParameterCount}");
        _output.WriteLine($"  vocab={vocabSize}, seq={seqLen}, embed={embedDim}, layers={numLayers}");

        // Build a token sequence
        var tokens = new Tensor<double>([seqLen]);
        for (int s = 0; s < seqLen; s++) tokens[s] = s % vocabSize;

        var logits = model.Forward(tokens);

        Assert.Equal(2, logits.Shape.Length);
        Assert.Equal(seqLen, logits.Shape[0]);
        Assert.Equal(vocabSize, logits.Shape[1]);

        double maxAbs = 0;
        for (int s = 0; s < seqLen; s++)
        {
            for (int v = 0; v < vocabSize; v++)
            {
                Assert.False(double.IsNaN(logits[s, v]), $"Logit[{s},{v}] is NaN");
                Assert.False(double.IsInfinity(logits[s, v]), $"Logit[{s},{v}] is Infinity");
                if (Math.Abs(logits[s, v]) > maxAbs) maxAbs = Math.Abs(logits[s, v]);
            }
        }

        _output.WriteLine($"Max absolute logit: {maxAbs:F4}");
        Assert.True(maxAbs > 1e-8, "Logits should have non-trivial magnitude");

        // Softmax of logits should be well-formed — for each position, probs sum to 1
        for (int s = 0; s < seqLen; s++)
        {
            double maxLogit = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
                if (logits[s, v] > maxLogit) maxLogit = logits[s, v];

            double sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
                sumExp += Math.Exp(logits[s, v] - maxLogit);

            Assert.True(sumExp > 0, $"Position {s}: softmax normalizer is 0");
            Assert.False(double.IsNaN(sumExp), $"Position {s}: softmax normalizer is NaN");
            Assert.False(double.IsInfinity(sumExp), $"Position {s}: softmax normalizer is Infinity");
        }
    }

    [Fact]
    public void HRELanguageModel_BatchedForward_ProducesCorrectShape()
    {
        const int vocabSize = 32;
        const int seqLen = 8;
        const int embedDim = 16;
        const int numLayers = 2;
        const int fftSize = 1024;
        const int batchSize = 3;

        var model = new HRELanguageModel<double>(
            vocabSize, seqLen, embedDim, numLayers, fftSize, seed: 42);

        var tokens = new Tensor<double>([batchSize, seqLen]);
        var rng = RandomHelper.CreateSecureRandom();
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < seqLen; s++)
                tokens[b, s] = rng.Next(vocabSize);

        var logits = model.Forward(tokens);

        Assert.Equal(3, logits.Shape.Length);
        Assert.Equal(batchSize, logits.Shape[0]);
        Assert.Equal(seqLen, logits.Shape[1]);
        Assert.Equal(vocabSize, logits.Shape[2]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int v = 0; v < vocabSize; v++)
                {
                    Assert.False(double.IsNaN(logits[b, s, v]));
                    Assert.False(double.IsInfinity(logits[b, s, v]));
                }
            }
        }
    }

    [Fact]
    public void HRELanguageModel_DifferentTokens_ProduceDifferentLogits()
    {
        const int vocabSize = 32;
        const int seqLen = 8;
        const int embedDim = 16;
        const int numLayers = 2;
        const int fftSize = 1024;

        var model = new HRELanguageModel<double>(
            vocabSize, seqLen, embedDim, numLayers, fftSize, seed: 42);

        var tokens1 = new Tensor<double>([seqLen]);
        var tokens2 = new Tensor<double>([seqLen]);
        for (int s = 0; s < seqLen; s++)
        {
            tokens1[s] = s % vocabSize;
            tokens2[s] = (s + 7) % vocabSize;
        }

        var logits1 = model.Forward(tokens1);
        var logits2 = model.Forward(tokens2);

        double diff = 0;
        for (int s = 0; s < seqLen; s++)
            for (int v = 0; v < vocabSize; v++)
                diff += Math.Abs(logits1[s, v] - logits2[s, v]);

        _output.WriteLine($"Total logit difference between inputs: {diff:F4}");
        Assert.True(diff > 1.0,
            $"Different input tokens should produce noticeably different logits, got diff={diff:F4}");
    }
}
