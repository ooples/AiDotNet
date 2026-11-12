using System;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks;

/// <summary>
/// An autoregressive decoder's self-attention is causal (Vaswani et al. 2017, §3.1); a decoder over a set of learned
/// queries is not: DETR's object queries (Carion et al. 2020, §3.2) and BLIP-2's Q-Former queries (Li et al. 2023,
/// §3.1) attend to each other in both directions.
/// </summary>
/// <remarks>The encoder memory is held fixed, so the only path from the last position to the first is the
/// self-attention.</remarks>
public sealed class TransformerDecoderCausalityTests
{
    private const int Positions = 3;
    private const int Width = 4;

    private static Tensor<double> Sequence(double lastPositionValue)
    {
        var x = new Tensor<double>(new[] { 1, Positions, Width });
        for (int p = 0; p < Positions; p++)
            for (int d = 0; d < Width; d++)
                x[0, p, d] = p == Positions - 1 ? lastPositionValue + 0.1 * d : 0.1 * (p + 1) + 0.05 * d;
        return x;
    }

    private static double[] FirstPosition(TransformerDecoderLayer<double> decoder, Tensor<double> input, Tensor<double> memory)
    {
        var output = decoder.Forward(input, memory);
        return Enumerable.Range(0, Width).Select(d => output[0, 0, d]).ToArray();
    }

    private static (double[] Before, double[] After) FirstPositionBeforeAndAfterChangingTheLast(bool causal)
    {
        var decoder = new TransformerDecoderLayer<double>(numHeads: 2, feedForwardDim: 8, sequenceLength: Positions,
            causal: causal);
        decoder.SetTrainingMode(false);
        var memory = Sequence(0.3);
        return (FirstPosition(decoder, Sequence(0.3), memory), FirstPosition(decoder, Sequence(-2.0), memory));
    }

    [Fact]
    public void CausalSelfAttention_KeepsLaterPositionsFromReachingEarlierOnes()
    {
        var (before, after) = FirstPositionBeforeAndAfterChangingTheLast(causal: true);
        Assert.Equal(before, after);
    }

    [Fact]
    public void BidirectionalSelfAttention_LetsEveryPositionSeeTheOthers()
    {
        var (before, after) = FirstPositionBeforeAndAfterChangingTheLast(causal: false);
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Max() > 1e-9,
            "The first position ignored a change to the last one, so the self-attention was masked.");
    }

    [Fact]
    public void Causality_SurvivesCloning()
    {
        var decoder = new TransformerDecoderLayer<double>(numHeads: 2, feedForwardDim: 8, sequenceLength: Positions,
            causal: false);
        decoder.Forward(Sequence(0.3), Sequence(0.3));
        var clone = (TransformerDecoderLayer<double>)decoder.Clone();
        Assert.False(clone.IsCausal);
    }
}
