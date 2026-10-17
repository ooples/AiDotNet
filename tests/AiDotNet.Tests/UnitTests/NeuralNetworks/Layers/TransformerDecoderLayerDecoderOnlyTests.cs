using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A <see cref="TransformerDecoderLayer{T}"/> given one input is a decoder-only block: self-attention, then the
/// feed-forward network, with no cross-attention sublayer (#2325). It used to cross-attend to its own input, an
/// unmasked second attention that let a position read the tokens after it even with causal self-attention.
/// </summary>
public class TransformerDecoderLayerDecoderOnlyTests
{
    private const int Sequence = 5, Width = 8;

    private static Tensor<double> Tokens(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(new[] { 1, Sequence, Width });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static TransformerDecoderLayer<double> Layer()
    {
        var layer = new TransformerDecoderLayer<double>(numHeads: 2, feedForwardDim: 16, sequenceLength: Sequence, causal: true);
        layer.SetTrainingMode(false);
        return layer;
    }

    [Fact]
    public void CausalDecoderOnlyBlock_DoesNotReadLaterTokens()
    {
        var layer = Layer();
        var tokens = Tokens(3);
        var before = layer.Forward(tokens);

        var changed = tokens.Clone();
        for (int f = 0; f < Width; f++) changed[0, Sequence - 1, f] += 5.0; // only the last token changes
        var after = layer.Forward(changed);

        for (int position = 0; position < Sequence - 1; position++)
        {
            for (int f = 0; f < Width; f++)
            {
                Assert.True(Math.Abs(before[0, position, f] - after[0, position, f]) <= 1e-12,
                    $"position {position} changed when only the last token did: the block reads tokens after it.");
            }
        }
    }

    [Fact]
    public void SingleInputForward_IsTheBlockWithoutCrossAttention()
    {
        var layer = Layer();
        var tokens = Tokens(4);
        var single = layer.Forward(tokens);
        var withoutCross = layer.ForwardWithoutCrossAttention(tokens);
        Assert.Equal(withoutCross.ToArray(), single.ToArray());
    }
}