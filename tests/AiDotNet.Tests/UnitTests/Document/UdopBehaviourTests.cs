using System;
using System.Linq;
using AiDotNet.Document.Options;
using AiDotNet.Document.VisionLanguage;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// Behavioural checks of the T5-based UDOP rebuild. A plausible wiring mistake would fail each one.
/// </summary>
public class UdopBehaviourTests
{
    private const int Vocab = 40;

    private static UDOP<double> Model() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: 4, inputWidth: 5, outputSize: Vocab),
        options: new UDOPOptions
        {
            ImageSize = 16,
            PatchSize = 8,
            MaxSequenceLength = 16,
            HiddenDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 2,
            NumHeads = 2,
            KeyValueDim = 8,
            FeedForwardDim = 32,
            VocabSize = Vocab,
            RelativeAttentionBuckets = 8,
            RelativeAttentionMaxDistance = 16,
            RelativeAttentionMaxDistance2D = 10,
            Max2DPositions = 16,
            NumLocationBins = 8,
            MaxGenerationLength = 5
        });

    // vocab 20, d 8, 2 heads of 4, d_ff 16, 1+1 blocks, 16px pages of 8px patches (a 2x2 grid), 8 buckets,
    // 1-D max distance 16, 2-D max distance 10, and max2DPositions as given.
    private static UdopTransformerLayer<double> Layer(int max2DPositions = 16) =>
        new(20, 8, 2, 4, 16, 1, 1, 16, 8, 8, 16, 10, max2DPositions);

    private static Tensor<double> Packed()
    {
        var t = new Tensor<double>(new[] { 4, 5 });
        for (int i = 0; i < 4; i++)
        {
            t[i, 0] = 3 + (7 * i);
            t[i, 1] = 100 + (200 * i); t[i, 2] = 120; t[i, 3] = 180 + (200 * i); t[i, 4] = 300;
        }
        return t;
    }

    private static Tensor<double> Page(int seed, int size = 16)
    {
        var rng = new Random(seed);
        var page = new Tensor<double>(new[] { 3, size, size });
        for (int i = 0; i < page.Length; i++) page[i] = rng.NextDouble();
        return page;
    }

    private static double RowDiff(Tensor<double> a, Tensor<double> b, int row) =>
        Enumerable.Range(0, a.Shape[1]).Sum(c => Math.Abs(a[row, c] - b[row, c]));

    [Fact]
    public void Predict_ReturnsStartTokenLogits()
    {
        using var model = Model();
        using var packed = Packed();
        using var logits = model.Predict(packed);
        Assert.Equal(new[] { 1, Vocab }, logits.Shape.ToArray());
    }

    [Fact]
    public void Decoder_IsCausal()
    {
        using var model = Model();
        using var packed = Packed();
        var a = model.DecoderLogits(packed, null, new[] { 0, 5, 6, 7 });
        var b = model.DecoderLogits(packed, null, new[] { 0, 5, 6, 30 });
        for (int row = 0; row < 3; row++)
            Assert.True(RowDiff(a, b, row) < 1e-9, $"row {row} changed when only the last decoder token changed: not causal");
        Assert.True(RowDiff(a, b, 3) > 1e-9, "The changed decoder token did not affect its own row: the decoder ignores its input.");
    }

    [Fact]
    public void PageImage_ReachesTheLogits()
    {
        using var model = Model();
        using var packed = Packed();
        using var first = Page(1);
        using var second = Page(2);
        var a = model.PredictDocument(packed, first);
        var b = model.PredictDocument(packed, second);
        Assert.True(RowDiff(a, b, 0) > 1e-8, "Two different pages gave the same logits: the visual stream is disconnected.");
    }

    [Fact]
    public void LayoutFusion_AddsThePatchUnderTheTokenBox_AndClaimsIt()
    {
        // One token centred in patch 0 of a 2x2 grid. Patch 0 is fused into the token and so is not appended:
        // the sequence is the token plus the three unclaimed patches. Changing only patch 0's pixels can then
        // reach the encoding through nothing but the fusion.
        using var layer = Layer();
        var box = new[] { new[] { 0.1, 0.1, 0.3, 0.3 } };
        using var page = Page(3);
        using var altered = page.Clone();
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++) altered[c, y, x] = 1 - page[c, y, x];

        var a = layer.Encode(new[] { 5 }, box, Batch(page));
        var b = layer.Encode(new[] { 5 }, box, Batch(altered));
        Assert.Equal(4, a.Shape[0]);
        Assert.True(RowDiff(a, b, 0) > 1e-8, "Changing the patch under the token's box left its encoding unchanged: no layout-induced fusion.");
    }

    [Fact]
    public void LayoutFusion_SkipsPromptTokens()
    {
        // An all-zero box marks a prompt token. Its centre still claims patch 0, but the reference zeroes the
        // fused patch, so patch 0 must reach nothing at all.
        using var layer = Layer();
        var box = new[] { new double[4] };
        using var page = Page(4);
        using var altered = page.Clone();
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++) altered[c, y, x] = 1 - page[c, y, x];

        var a = layer.Encode(new[] { 5 }, box, Batch(page));
        var b = layer.Encode(new[] { 5 }, box, Batch(altered));
        Assert.Equal(4, a.Shape[0]);
        for (int row = 0; row < 4; row++)
            Assert.True(RowDiff(a, b, row) < 1e-9, $"row {row} saw the claimed patch under a prompt token, which the reference skips.");
    }

    [Fact]
    public void Encoder_IsOrderSensitiveThroughTheOneDimensionalBias()
    {
        // T5 has no absolute positions. With identical boxes the 2-D terms are equal for every pair, so only
        // the 1-D bias can tell [a, b, c] from [b, a, c] at row 2.
        using var layer = Layer();
        var boxes = Enumerable.Range(0, 3).Select(_ => new[] { 0.2, 0.2, 0.4, 0.4 }).ToArray();
        var abc = layer.Encode(new[] { 3, 11, 17 }, boxes, null);
        var bac = layer.Encode(new[] { 11, 3, 17 }, boxes, null);
        Assert.True(RowDiff(abc, bac, 2) > 1e-8, "Reordering the context left the last row unchanged: the 1-D relative bias is not applied.");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Encoder_AttendsByLayoutDistance(bool horizontal)
    {
        // Max2DPositions = 2 puts every coordinate below 1 in cell 0, so moving token 1's box does not change
        // any cell embedding. Moving it along one axis then changes only that axis's 2-D relative bias.
        using var layer = Layer(max2DPositions: 2);
        var near = new[] { new[] { 0.1, 0.1, 0.2, 0.2 }, new[] { 0.1, 0.1, 0.2, 0.2 }, new[] { 0.1, 0.1, 0.2, 0.2 } };
        var far = near.Select(b => (double[])b.Clone()).ToArray();
        far[1] = horizontal ? new[] { 0.7, 0.1, 0.8, 0.2 } : new[] { 0.1, 0.7, 0.2, 0.8 };
        var a = layer.Encode(new[] { 3, 11, 17 }, near, null);
        var b = layer.Encode(new[] { 3, 11, 17 }, far, null);
        Assert.True(RowDiff(a, b, 0) > 1e-8,
            $"Moving another token {(horizontal ? "horizontally" : "vertically")} left row 0 unchanged: that 2-D relative bias is not applied.");
    }

    [Fact]
    public void TeacherForcedTraining_LowersTheLoss()
    {
        using var model = Model();
        using var packed = Packed();
        int[] target = { 9, 14, 3, 1 };
        // Measured here from the logits, not read back from the model's own loss: a loss with the wrong sign
        // also "decreases" while it drives the target's likelihood down.
        double CrossEntropy()
        {
            var logits = model.DecoderLogits(packed, null, new[] { 0, 9, 14, 3 });
            double total = 0;
            for (int t = 0; t < target.Length; t++)
            {
                double max = Enumerable.Range(0, Vocab).Max(v => logits[t, v]);
                double logSum = max + Math.Log(Enumerable.Range(0, Vocab).Sum(v => Math.Exp(logits[t, v] - max)));
                total += logSum - logits[t, target[t]];
            }
            return total / target.Length;
        }

        double first = CrossEntropy();
        for (int step = 0; step < 10; step++) model.TrainDocument(packed, null, target);
        double last = CrossEntropy();
        Assert.False(double.IsNaN(last) || double.IsInfinity(last));
        Assert.True(last < first, $"10 teacher-forced steps did not lower the target's cross-entropy: {first} -> {last}");
    }

    [Fact]
    public void GenerateTokens_StopsWithinTheLimit()
    {
        using var model = Model();
        using var packed = Packed();
        using var page = Page(5);
        var tokens = model.GenerateTokens(packed, page);
        Assert.InRange(tokens.Length, 1, 5);
        for (int i = 0; i < tokens.Length; i++) Assert.InRange(tokens[i], 0, Vocab - 1);
    }

    [Fact]
    public void ClassifyDocument_ScoresEveryCategory()
    {
        using var model = Model();
        using var page = Page(6);
        var result = model.ClassifyDocument(page, model.AvailableCategories.Count);
        Assert.Equal(model.AvailableCategories.Count, result.TopPredictions.Count);
        Assert.InRange(result.TopPredictions.Sum(p => p.Score), 1 - 1e-9, 1 + 1e-9);
        Assert.Contains(result.PredictedCategory, model.AvailableCategories);
    }

    private static Tensor<double> Batch(Tensor<double> page)
    {
        var batched = new Tensor<double>(new[] { 1, page.Shape[0], page.Shape[1], page.Shape[2] });
        for (int i = 0; i < page.Length; i++) batched[i] = page[i];
        return batched;
    }
}
