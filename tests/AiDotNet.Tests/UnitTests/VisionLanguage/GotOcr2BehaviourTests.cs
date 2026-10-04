using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Document;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Behavioural checks of the SAM ViTDet + Qwen GOT-OCR2 rebuild. A plausible wiring mistake would fail each one.
/// </summary>
public class GotOcr2BehaviourTests
{
    private const int Vocab = 64;

    private static GOTOCR2Options Options() => new()
    {
        ImageSize = 128,
        PatchSize = 16,
        VisionDim = 16,
        NumVisionLayers = 2,
        NumHeads = 2,
        VisionMlpDim = 32,
        WindowSize = 4,
        GlobalAttentionEvery = 2,
        NeckChannels = 8,
        DecoderDim = 16,
        NumDecoderLayers = 2,
        DecoderHeads = 2,
        DecoderKeyValueHeads = 1,
        DecoderFeedForwardDim = 32,
        VocabSize = Vocab,
        EndOfTextTokenId = 58,
        ImStartTokenId = 59,
        ImEndTokenId = 60,
        MaxSequenceLength = 160,
        MaxGenerationLength = 5,
        DropoutRate = 0.0
    };

    private static GOTOCR2<double> Model() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration, inputHeight: 128, inputWidth: 128, inputDepth: 3, outputSize: Vocab),
        Options());

    // A 32 px page of 8 px patches: a 4x4 grid. Block 0 uses 2x2 windows; block 1 (every 2nd) is global.
    private static SamViTDetEncoderLayer<double> Encoder() => new(32, 8, 8, 2, 2, 16, 2, 2, 4);

    private static Tensor<double> Image(int seed, int size = 128)
    {
        var rng = new Random(seed);
        var image = new Tensor<double>(new[] { 3, size, size });
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
    public void Predict_ReturnsTheFirstAnswerTokenLogits()
    {
        using var model = Model();
        using var image = Image(1);
        using var logits = model.Predict(image);
        Assert.Equal(new[] { 1, Vocab }, logits.Shape.ToArray());
    }

    [Fact]
    public void Prompt_HoldsOneImageSlotPerImageToken_InsideImgTags()
    {
        using var model = Model();
        var options = Options();
        var (ids, imageStart) = model.Prompt("OCR: ");
        Assert.Equal(4, model.ImageTokenCount);
        Assert.Equal(options.ImageStartTokenId, ids[imageStart - 1]);
        for (int i = 0; i < 4; i++) Assert.Equal(options.ImagePadTokenId, ids[imageStart + i]);
        Assert.Equal(options.ImageEndTokenId, ids[imageStart + 4]);
        Assert.Equal(options.ImStartTokenId, ids[0]);
        Assert.Equal(4, ids.Count(id => id == options.ImagePadTokenId));
    }

    [Fact]
    public void Decoder_IsCausal()
    {
        // Row t is predicted from answer[0..t-1]. Changing answer[2] can move row 3 and nothing before it.
        using var model = Model();
        using var image = Image(2);
        var a = model.PredictTokens(image, new[] { 5, 6, 7, 8 });
        var b = model.PredictTokens(image, new[] { 5, 6, 30, 8 });
        for (int row = 0; row < 3; row++)
            Assert.True(RowDiff(a, b, row) < 1e-9, $"row {row} changed when only a later answer token changed: not causal");
        Assert.True(RowDiff(a, b, 3) > 1e-9, "The changed answer token did not reach the next row.");
    }

    [Fact]
    public void ImageReachesTheDecoder()
    {
        using var model = Model();
        using var first = Image(3);
        using var second = Image(4);
        var a = model.PredictTokens(first, new[] { 5 });
        var b = model.PredictTokens(second, new[] { 5 });
        Assert.True(RowDiff(a, b, 0) > 1e-8, "Two different pages gave the same logits: the image slots are never filled.");
    }

    [Fact]
    public void WindowedBlock_StaysInsideItsWindow_AndGlobalBlock_DoesNot()
    {
        using var encoder = Encoder();
        using var tokens = Random2D(16, 8, 7);
        using var altered = tokens.Clone();
        for (int c = 0; c < 8; c++) altered[0, c] += 1.0;
        // Token 0 is (0,0) in window (0,0). Token 15 is (3,3) in window (1,1).
        var windowedA = encoder.Attention(0, tokens);
        var windowedB = encoder.Attention(0, altered);
        Assert.True(RowDiff(windowedA, windowedB, 15) < 1e-12, "Block 0 let token (3,3) see token (0,0) across windows.");
        Assert.True(RowDiff(windowedA, windowedB, 1) > 1e-9, "Block 0 hid token (0,0) from (0,1) in its own window.");
        var globalA = encoder.Attention(1, tokens);
        var globalB = encoder.Attention(1, altered);
        Assert.True(RowDiff(globalA, globalB, 15) > 1e-9, "Block 1 is global yet token (3,3) did not see token (0,0).");
    }

    [Fact]
    public void RelativeBias_IsTheDecomposedHeightPlusWidthTerm()
    {
        // bias[(qh,qw),(kh,kw)] = q . Rh[qh - kh + S - 1] + q . Rw[qw - kw + S - 1] (Li et al. 2022).
        using var encoder = Encoder();
        var rng = new Random(13);
        Tensor<double> Fill(string field)
        {
            var table = (Tensor<double>)(typeof(SamViTDetEncoderLayer<double>)
                .GetField(field, BindingFlags.NonPublic | BindingFlags.Instance)
                ?.GetValue(encoder) ?? throw new InvalidOperationException(field));
            for (int i = 0; i < table.Length; i++) table[i] = rng.NextDouble() - 0.5;
            return table;
        }
        var rh = Fill("_relativeGlobalHeight");
        var rw = Fill("_relativeGlobalWidth");
        const int s = 4, hd = 4, batch = 2;
        var q = new Tensor<double>(new[] { batch, s * s, hd });
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextDouble() - 0.5;
        var bias = encoder.RelativeBias(1, q, s, batch);
        for (int b = 0; b < batch; b++)
            for (int qh = 0; qh < s; qh++)
                for (int qw = 0; qw < s; qw++)
                    for (int kh = 0; kh < s; kh++)
                        for (int kw = 0; kw < s; kw++)
                        {
                            double expected = 0;
                            for (int c = 0; c < hd; c++)
                                expected += q[b, (qh * s) + qw, c] * (rh[0, qh - kh + s - 1, c] + rw[0, qw - kw + s - 1, c]);
                            Assert.Equal(expected, bias[b, (qh * s) + qw, (kh * s) + kw], 10);
                        }
    }

    [Fact]
    public void Projector_ReducesTheGridFourTimesPerSide()
    {
        using var projector = new GotOcr2ProjectorLayer<double>(4, 6, 8);
        var rng = new Random(9);
        using var map = new Tensor<double>(new[] { 4, 8, 8 });
        for (int i = 0; i < map.Length; i++) map[i] = rng.NextDouble();
        using var tokens = projector.Forward(map);
        Assert.Equal(new[] { 4, 6 }, tokens.Shape.ToArray());
    }

    [Fact]
    public void TeacherForcedTraining_LowersTheCrossEntropy()
    {
        using var model = Model();
        using var image = Image(6);
        int[] target = { 9, 14, 3, 60 };
        // Measured from the logits, not the model's reported loss: a sign-flipped loss also "decreases".
        double CrossEntropy()
        {
            var logits = model.PredictTokens(image, target);
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
        for (int step = 0; step < 10; step++) model.TrainOcr(image, target);
        double last = CrossEntropy();
        Assert.False(double.IsNaN(last) || double.IsInfinity(last));
        Assert.True(last < first, $"10 teacher-forced steps did not lower the answer's cross-entropy: {first} -> {last}");
    }

    [Fact]
    public void GenerateFromImage_StopsWithinTheLimit()
    {
        using var model = Model();
        using var image = Image(7);
        var tokens = model.GenerateFromImage(image);
        Assert.InRange(tokens.Length, 1, 5);
        for (int i = 0; i < tokens.Length; i++) Assert.InRange(tokens[i], 0, Vocab - 1);
    }
}
