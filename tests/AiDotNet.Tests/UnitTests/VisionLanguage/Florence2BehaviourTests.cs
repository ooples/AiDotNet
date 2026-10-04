using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Behavioural checks of the DaViT + BART Florence-2 rebuild. A plausible wiring mistake would fail each one.
/// </summary>
public class Florence2BehaviourTests
{
    private const int Vocab = 64;
    private const int LocationBins = 16;
    private const int FirstLocation = Vocab - LocationBins;

    private static Florence2<double> Model() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration, inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: Vocab),
        new Florence2Options
        {
            ImageSize = 64,
            VisionBaseDim = 8,
            VisionBaseHeads = 1,
            VisionThirdStageDepth = 1,
            WindowSize = 2,
            FfnMultiplier = 2,
            EmbeddingDim = 16,
            TextDim = 16,
            NumLayers = 1,
            NumDecoderLayers = 2,
            NumHeads = 2,
            TextFeedForwardDim = 32,
            VocabSize = Vocab,
            MaxTextPositions = 64,
            NumLocationBins = LocationBins,
            MaxOutputTokens = 6,
            DropoutRate = 0.0
        });

    private static Tensor<double> Image(int seed)
    {
        var rng = new Random(seed);
        var image = new Tensor<double>(new[] { 3, 64, 64 });
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble();
        return image;
    }

    private static Tensor<double> Random2D(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static double RowDiff(Tensor<double> a, Tensor<double> b, int row) =>
        Enumerable.Range(0, a.Shape[1]).Sum(c => Math.Abs(a[row, c] - b[row, c]));

    [Fact]
    public void Predict_ReturnsStartTokenLogits()
    {
        using var model = Model();
        using var image = Image(1);
        using var logits = model.Predict(image);
        Assert.Equal(new[] { 1, Vocab }, logits.Shape.ToArray());
    }

    [Fact]
    public void Decoder_IsCausal()
    {
        using var model = Model();
        using var image = Image(2);
        var a = model.PredictTokens(image, new[] { 2, 5, 6, 7 });
        var b = model.PredictTokens(image, new[] { 2, 5, 6, 30 });
        for (int row = 0; row < 3; row++)
            Assert.True(RowDiff(a, b, row) < 1e-9, $"row {row} changed when only the last decoder token changed: not causal");
        Assert.True(RowDiff(a, b, 3) > 1e-9, "The changed decoder token did not affect its own row.");
    }

    [Fact]
    public void ImageReachesTheDecoder()
    {
        using var model = Model();
        using var first = Image(3);
        using var second = Image(4);
        var a = model.PredictTokens(first, new[] { 2 });
        var b = model.PredictTokens(second, new[] { 2 });
        Assert.True(RowDiff(a, b, 0) > 1e-8, "Two different images gave the same logits: the image tokens never reach BART.");
    }

    [Fact]
    public void TaskPromptChangesTheAnswer()
    {
        using var model = Model();
        using var image = Image(5);
        var ocr = model.PredictTokens(image, new[] { 2 }, Florence2Task.Ocr);
        var caption = model.PredictTokens(image, new[] { 2 }, Florence2Task.Caption);
        Assert.True(RowDiff(ocr, caption, 0) > 1e-8, "The OCR and caption prompts gave the same logits: the task prompt is not encoded.");
    }

    [Fact]
    public void WindowAttention_StaysInsideItsWindow()
    {
        // A 4x4 map in 2x2 windows. Token (0,0) is in window (0,0); tokens (2,2)..(3,3) form window (1,1).
        using var davit = new DaViTLayer<double>(64, 4, 1, 1, 2, 2);
        using var tokens = Random2D(16, 4, 7);
        using var altered = tokens.Clone();
        for (int c = 0; c < 4; c++) altered[0, c] += 1.0;
        var a = davit.WindowAttention(0, tokens, 4, 4, 1);
        var b = davit.WindowAttention(0, altered, 4, 4, 1);
        foreach (int other in new[] { 10, 11, 14, 15 })
            Assert.True(RowDiff(a, b, other) < 1e-12, $"token {other} in another window saw token 0: attention is not windowed");
        Assert.True(RowDiff(a, b, 1) > 1e-9, "Token (0,1) did not see token (0,0) in its own window.");
    }

    [Fact]
    public void ChannelAttention_MixesTheWholeMap()
    {
        // Channel attention scores channel against channel, summing over EVERY position, so a change in window
        // (0,0) reaches window (1,1). Window attention cannot do that (see the test above).
        using var davit = new DaViTLayer<double>(64, 4, 1, 1, 2, 2);
        using var tokens = Random2D(16, 4, 8);
        using var altered = tokens.Clone();
        for (int c = 0; c < 4; c++) altered[0, c] += 1.0;
        var a = davit.ChannelAttention(1, tokens, 1);
        var b = davit.ChannelAttention(1, altered, 1);
        Assert.True(RowDiff(a, b, 15) > 1e-9, "Token (3,3) did not see a change at (0,0): channel attention is not global.");
    }
    [Fact]
    public void Projector_PrependsTheMeanOverPositions()
    {
        // Position embeddings are added before the mean, so mean(x + pos) = mean(x) + mean(pos): swapping two
        // positions' features leaves the first token alone and moves the per-position tokens.
        using var projector = new Florence2ProjectorLayer<double>(4, 6, 2);
        var rng = new Random(9);
        using var map = new Tensor<double>(new[] { 4, 2, 2 });
        for (int i = 0; i < map.Length; i++) map[i] = rng.NextDouble() - 0.5;
        using var swapped = map.Clone();
        for (int c = 0; c < 4; c++) { swapped[c, 0, 0] = map[c, 1, 1]; swapped[c, 1, 1] = map[c, 0, 0]; }
        var a = projector.Forward(map);
        var b = projector.Forward(swapped);
        Assert.Equal(new[] { 5, 6 }, a.Shape.ToArray());
        Assert.True(RowDiff(a, b, 0) < 1e-9, "The first token changed when positions were swapped: it is not the mean over positions.");
        Assert.True(RowDiff(a, b, 1) > 1e-9, "Position (0,0)'s token ignored its own features.");
    }

    [Fact]
    public void ParseAnswer_DequantisesLocationTokens()
    {
        using var model = Model();
        // label, then x0 = bin 2, y0 = bin 4, x1 = bin 10, y1 = bin 12, then EOS. Bins dequantise to
        // (bin + 0.5) / 16 of the side.
        var result = model.ParseAnswer(new[] { 5, FirstLocation + 2, FirstLocation + 4, FirstLocation + 10, FirstLocation + 12, 2, 9 }, 64, 32);
        var region = Assert.Single(result.Regions);
        Assert.Equal(2.5 / 16 * 64, region.X0, 9);
        Assert.Equal(4.5 / 16 * 32, region.Y0, 9);
        Assert.Equal(10.5 / 16 * 64, region.X1, 9);
        Assert.Equal(12.5 / 16 * 32, region.Y1, 9);
    }

    [Fact]
    public void TeacherForcedTraining_LowersTheCrossEntropy()
    {
        using var model = Model();
        using var image = Image(6);
        int[] target = { 9, 14, FirstLocation + 3, 2 };
        // Measured from the logits, not from the model's reported loss: a sign-flipped loss also "decreases".
        double CrossEntropy()
        {
            var logits = model.PredictTokens(image, new[] { 2, 9, 14, FirstLocation + 3 });
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
        for (int step = 0; step < 10; step++) model.TrainCaption(image, target);
        double last = CrossEntropy();
        Assert.False(double.IsNaN(last) || double.IsInfinity(last));
        Assert.True(last < first, $"10 teacher-forced steps did not lower the target's cross-entropy: {first} -> {last}");
    }

    [Fact]
    public void GenerateFromImage_StopsWithinTheLimit()
    {
        using var model = Model();
        using var image = Image(7);
        var tokens = model.GenerateFromImage(image, Florence2Task.Caption);
        Assert.InRange(tokens.Length, 1, 6);
        for (int i = 0; i < tokens.Length; i++) Assert.InRange(tokens[i], 0, Vocab - 1);
    }
}
