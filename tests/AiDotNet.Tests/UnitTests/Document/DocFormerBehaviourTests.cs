using System;
using System.Linq;
using AiDotNet.Document.LayoutAware;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// Behavioural checks of DocFormer's multi-modal self-attention rebuild. A plausible wiring mistake would fail
/// each of them.
/// </summary>
public class DocFormerBehaviourTests
{
    private static DocFormer<double> Model() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: 8, inputWidth: 5, outputSize: 4),
        options: new DocFormerOptions
        {
            NumClasses = 4,
            ImageSize = 64,
            MaxSequenceLength = 8,
            HiddenDim = 32,
            NumLayers = 2,
            NumHeads = 4,
            VocabSize = 50,
            SpatialDim = 4
        });

    private static Tensor<double> Packed()
    {
        var t = new Tensor<double>(new[] { 6, 5 });
        for (int i = 0; i < 6; i++)
        {
            t[i, 0] = 3 + (5 * i);
            t[i, 1] = 20 * i; t[i, 2] = 10; t[i, 3] = (20 * i) + 15; t[i, 4] = 30;
        }
        return t;
    }

    private static Tensor<double> Page(int seed)
    {
        var rng = new Random(seed);
        var page = new Tensor<double>(new[] { 3, 64, 64 });
        for (int i = 0; i < page.Length; i++) page[i] = rng.NextDouble();
        return page;
    }

    [Fact]
    public void PageImage_ReachesTheTokenLogits()
    {
        using var model = Model();
        using var packed = Packed();
        using var first = Page(1);
        using var second = Page(2);
        using var a = model.PredictDocument(packed, first);
        using var b = model.PredictDocument(packed, second);
        Assert.Equal(new[] { 6, 4 }, a.Shape.ToArray());
        double diff = Enumerable.Range(0, a.Length).Sum(i => Math.Abs(a[i] - b[i]));
        Assert.True(diff > 1e-8, "Two different pages gave the same token logits: the visual stream is disconnected.");
    }

    [Fact]
    public void Encoder_IsOrderSensitiveThroughRelativePositions()
    {
        // With the visual and spatial streams zero, attention without position terms is permutation-invariant
        // over its keys. Row 2 of [a, b, c] and of [b, a, c] then sees the same query and the same key set.
        // Only the relative 1-D terms can tell the two orders apart.
        using var encoder = new DocFormerEncoderLayer<double>(8, 2, 16, 2, 1);
        var rng = new Random(7);
        var rows = Enumerable.Range(0, 3).Select(_ => Enumerable.Range(0, 8).Select(__ => rng.NextDouble() - 0.5).ToArray()).ToArray();
        Tensor<double> Seq(params int[] order)
        {
            var t = new Tensor<double>(new[] { 3, 8 });
            for (int i = 0; i < 3; i++) for (int d = 0; d < 8; d++) t[i, d] = rows[order[i]][d];
            return t;
        }
        using var abc = Seq(0, 1, 2);
        using var bac = Seq(1, 0, 2);
        using var first = encoder.Forward(abc);
        using var second = encoder.Forward(bac);
        double diff = Enumerable.Range(0, 8).Sum(d => Math.Abs(first[2, d] - second[2, d]));
        Assert.True(diff > 1e-8, "Reordering the context left the last row unchanged: the relative position terms are not applied.");
    }

    [Fact]
    public void SpatialScores_SteerTextAttention()
    {
        // Text fixed, visual streams zero. Swap the spatial features of rows 0 and 1 and keep row 2's own. Row
        // 2's skip (t2 + ts2) is then unchanged, and so are its text and relative scores, so only the spatial
        // q/k scores can move its output. Without them row 2 would be identical.
        using var encoder = new DocFormerEncoderLayer<double>(8, 2, 16, 2, 1);
        var rng = new Random(11);
        using var text = new Tensor<double>(new[] { 3, 8 });
        using var zero = new Tensor<double>(new[] { 3, 8 });
        for (int i = 0; i < text.Length; i++) text[i] = rng.NextDouble() - 0.5;
        var spatial = Enumerable.Range(0, 3).Select(_ => Enumerable.Range(0, 8).Select(__ => 2 * (rng.NextDouble() - 0.5)).ToArray()).ToArray();
        Tensor<double> Spatial(params int[] order)
        {
            var t = new Tensor<double>(new[] { 3, 8 });
            for (int i = 0; i < 3; i++) for (int d = 0; d < 8; d++) t[i, d] = spatial[order[i]][d];
            return t;
        }
        using var original = Spatial(0, 1, 2);
        using var swapped = Spatial(1, 0, 2);
        using var first = encoder.Forward(text, zero, original, zero);
        using var second = encoder.Forward(text, zero, swapped, zero);
        double diff = Enumerable.Range(0, 8).Sum(d => Math.Abs(first[2, d] - second[2, d]));
        Assert.True(diff > 1e-8, "Swapping other tokens' spatial features left row 2 unchanged: the spatial attention scores are not applied.");
    }
}
