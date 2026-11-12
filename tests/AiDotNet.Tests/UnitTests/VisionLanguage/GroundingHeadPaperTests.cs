using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Encoders;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Reference checks for the grounding heads and SigLIP 2 heads rebuilt to their papers: Grounding DINO's query decoder
/// (Liu et al. 2023), OWL-ViT / OWLv2's per-token heads (Minderer et al. 2022, 2023), coordinate-token grounding (Shikra,
/// Ferret), Groma's region tokens (Ma et al. 2024), and SigLIP 2's LocCa decoder and masked prediction (Tschannen et al.
/// 2025).
/// </summary>
public class GroundingHeadPaperTests
{
    private const double Tolerance = 1e-9;

    private static IEngine Engine => AiDotNetEngine.Current;

    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<double> Matrix(int rows, int columns, params double[] values)
        => new(new[] { rows, columns }, new Vector<double>(values));

    [Fact]
    public void QueryDecoder_SineEmbedding_IsDinoGenSineEmbedForPosition()
    {
        var head = new GroundingQueryDecoder<double>(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, 2, 16, 1, 1, 3, 32, 4));
        var boxes = Matrix(2, 4, 0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.05, 0.6);

        var embedding = head.SineEmbedding(boxes);

        // Width 8: per coordinate, sin then cos of 2*pi*v / 10000^(2i/4) for i = 0, 1.
        Assert.Equal(new[] { 2, 16 }, embedding.Shape.ToArray());
        var frequencies = new[] { 1.0, Math.Pow(10000, 0.5) };
        for (int k = 0; k < 2; k++)
            for (int c = 0; c < 4; c++)
                for (int i = 0; i < 2; i++)
                {
                    double angle = 2 * Math.PI * boxes[k, c] / frequencies[i];
                    Assert.Equal(Math.Sin(angle), embedding[k, 4 * c + i], Tolerance);
                    Assert.Equal(Math.Cos(angle), embedding[k, 4 * c + 2 + i], Tolerance);
                }
    }

    [Fact]
    public void QueryDecoder_KeepsTheQueriesTheSelectionAllows_WithUnitBoxesAndTextLogits()
    {
        var head = new GroundingQueryDecoder<double>(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, 2, 16, 1, 2, 3, 32, 4));

        var prediction = head.Forward(Random(1, 6, 5), new[] { 3, 7, 11 });

        Assert.Equal(new[] { 3, 4 }, prediction.Boxes.Shape.ToArray());
        Assert.Equal(new[] { 3, 3 }, prediction.Logits.Shape.ToArray());
        var selected = prediction.Selected.ToArray();
        Assert.Equal(3, selected.Distinct().Count());
        Assert.All(selected, s => Assert.InRange(s, 0, 5));
        Assert.All(prediction.Boxes.ToArray(), v => Assert.True(v > 0 && v < 1));
    }

    [Fact]
    public void QueryDecoder_ClassAgnostic_ScoresQueriesByObjectness()
    {
        var head = new GroundingQueryDecoder<double>(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, 2, 16, 0, 1, 10, 0, 1));

        var prediction = head.Forward(Random(2, 4, 5), null);

        // Ten queries asked for, four tokens to select from.
        Assert.Equal(new[] { 4, 4 }, prediction.Boxes.Shape.ToArray());
        Assert.Equal(new[] { 4, 1 }, prediction.Logits.Shape.ToArray());
    }

    [Fact]
    public void OwlHead_BoxBias_IsTheReferenceComputeBoxBias()
    {
        var head = new OwlDetectionHead<double>(Engine, new List<ILayer<double>>(), 8,
            new OwlDetectionHeadConfiguration(8, 8, 1, 2, 32, 4, Objectness: false));

        var bias = head.BoxBias(2, 3);

        static double Logit(double x) => Math.Log(x + 1e-4) - Math.Log(1 - x + 1e-4);
        Assert.Equal(new[] { 6, 4 }, bias.Shape.ToArray());
        for (int row = 0; row < 2; row++)
            for (int column = 0; column < 3; column++)
            {
                int token = row * 3 + column;
                Assert.Equal(Logit((column + 1) / 3.0), bias[token, 0], Tolerance);
                Assert.Equal(Logit((row + 1) / 2.0), bias[token, 1], Tolerance);
                Assert.Equal(Logit(1 / 3.0), bias[token, 2], Tolerance);
                Assert.Equal(Logit(1 / 2.0), bias[token, 3], Tolerance);
            }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OwlHead_PredictsOneBoxAndLogitPerToken_AndOwlv2Objectness(bool objectness)
    {
        var head = new OwlDetectionHead<double>(Engine, new List<ILayer<double>>(), 8,
            new OwlDetectionHeadConfiguration(6, 8, 1, 2, 32, 4, objectness));

        var query = head.EmbedQuery(new[] { 4, 9, 2 });
        var prediction = head.Forward(Random(3, 9, 8), query);

        Assert.Equal(new[] { 1, 6 }, query.Shape.ToArray());
        Assert.Equal(new[] { 9, 4 }, prediction.Boxes.Shape.ToArray());
        Assert.Equal(new[] { 9, 1 }, prediction.Logits.Shape.ToArray());
        Assert.Equal(objectness, prediction.Objectness is not null);
        if (prediction.Objectness is not null) Assert.Equal(new[] { 9 }, prediction.Objectness.Shape.ToArray());
    }

    [Fact]
    public void CoordinateDecoder_AllowsOnlyWellFormedBoxes()
    {
        // Bins 5: tokens 0-4 are coordinates, 5 separates boxes, 6 ends the answer.
        var decoder = new CoordinateTokenDecoder<double>(Engine, new List<ILayer<double>>(),
            new CoordinateTokenDecoderConfiguration(8, 32, 4, Bins: 5, Divisor: 4));

        Assert.Equal(new double[] { 1, 1, 1, 1, 1, 0, 1 }, decoder.Allowed(Array.Empty<int>(), 2).ToArray());
        Assert.Equal(new double[] { 1, 1, 1, 1, 1, 0, 0 }, decoder.Allowed(new[] { 1 }, 2).ToArray());
        Assert.Equal(new double[] { 1, 1, 1, 1, 1, 0, 0 }, decoder.Allowed(new[] { 1, 2, 3 }, 2).ToArray());
        Assert.Equal(new double[] { 0, 0, 0, 0, 0, 1, 1 }, decoder.Allowed(new[] { 1, 2, 3, 4 }, 2).ToArray());
        Assert.Equal(new double[] { 1, 1, 1, 1, 1, 0, 1 }, decoder.Allowed(new[] { 1, 2, 3, 4, 5 }, 2).ToArray());
        Assert.Equal(new double[] { 0, 0, 0, 0, 0, 0, 1 }, decoder.Allowed(new[] { 1, 2, 3, 4, 5, 0, 1, 2, 3, 5 }, 2).ToArray());
    }

    [Fact]
    public void CoordinateDecoder_WritesBoxesAsCoordinateTokens_ScoredByTheirProbability()
    {
        var decoder = new CoordinateTokenDecoder<double>(Engine, new List<ILayer<double>>(),
            new CoordinateTokenDecoderConfiguration(8, 32, 4, Bins: 5, Divisor: 4));

        // A stand-in language model that passes its input through.
        var (boxes, scores) = decoder.Ground(Random(4, 3, 8), new[] { 1, 2 }, rows => rows, maxBoxes: 3);

        int count = scores.Length;
        Assert.InRange(count, 0, 3);
        Assert.Equal(new[] { count, 4 }, boxes.Shape.ToArray());
        // Every coordinate is a bin over the divisor: 0, 0.25, ..., 1.
        Assert.All(boxes.ToArray(), v => Assert.Equal(0.0, v * 4 - Math.Round(v * 4), Tolerance));
        Assert.All(scores.ToArray(), s => Assert.InRange(s, 0.0, 1.0));
    }

    [Fact]
    public void Groma_PoolsTheTokensInsideEachBox_OrTheNearestWhenNoneIs()
    {
        var grounder = new GromaRegionGrounder<double>(Engine, new List<ILayer<double>>(),
            new GromaRegionGrounderConfiguration(8, 8, 2, 1, 3, 32, 4));
        // A 2 x 2 grid: token centres (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75).
        var boxes = Matrix(3, 4,
            0.0, 0.0, 1.0, 0.5,     // the top row
            0.6, 0.6, 0.7, 0.7,     // no centre; nearest to (0.65, 0.65) is token 3
            0.0, 0.0, 1.0, 1.0);    // everything

        var weights = grounder.PoolingWeights(boxes, 2, 2);

        Assert.Equal(new[] { 0.5, 0.5, 0, 0, 0, 0, 0, 1, 0.25, 0.25, 0.25, 0.25 }, weights.ToArray());
    }

    [Fact]
    public void Ranked_KeepsTheStrongestBoxesAboveTheThreshold_HighestFirst()
    {
        var boxes = Matrix(4, 4,
            0.0, 0.0, 0.1, 0.1,
            0.2, 0.2, 0.3, 0.3,
            0.4, 0.4, 0.5, 0.5,
            0.6, 0.6, 0.7, 0.7);
        var scores = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.2, 0.9, 0.5, 0.95 }));

        var rows = GroundingDetections<double>.Ranked(Engine, boxes, scores, 0.3, 2, 0.5);

        Assert.Equal(new[] { 0.6, 0.6, 0.7, 0.7, 0.95, 0.2, 0.2, 0.3, 0.3, 0.9 }, rows.ToArray());
    }

    [Fact]
    public void DinoX_GroundText_ReturnsScoredBoxesHighestFirst()
    {
        using var model = new DINOX<double>(
            new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ObjectDetection, inputHeight: 8, inputWidth: 8, inputDepth: 3, outputSize: 4),
            new DINOXOptions
            {
                ImageSize = 8, VisionDim = 16, DecoderDim = 16, NumVisionLayers = 1, NumDecoderLayers = 1, NumHeads = 2,
                VocabSize = 512, MaxSequenceLength = 8, NumQueryPositions = 4, MaxDetections = 4,
                ConfidenceThreshold = 0.0, DropoutRate = 0.0,
            });
        var image = Random(5, 3, 8, 8);

        var rows = model.GroundText(image, "a cat").ToArray();

        Assert.True(rows.Length >= 5 && rows.Length % 5 == 0);
        for (int r = 0; r < rows.Length / 5; r++)
        {
            Assert.True(rows[5 * r + 2] > rows[5 * r] && rows[5 * r + 3] > rows[5 * r + 1]);
            Assert.InRange(rows[5 * r + 4], 0.0, 1.0);
            if (r > 0) Assert.True(rows[5 * r + 4] <= rows[5 * (r - 1) + 4]);
        }
    }

    private static SigLIP2<double> SmallSigLIP2() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 16),
        new SigLIP2Options
        {
            ImageSize = 32,
            PatchSize = 16,
            VisionEmbeddingDim = 16,
            TextEmbeddingDim = 16,
            ProjectionDim = 16,
            NumVisionLayers = 1,
            NumTextLayers = 1,
            NumVisionHeads = 2,
            NumTextHeads = 2,
            CaptioningDecoderDim = 16,
            NumCaptioningDecoderLayers = 1,
            NumCaptioningDecoderHeads = 2,
            MimDecoderDim = 16,
            NumMimDecoderLayers = 1,
            VocabSize = 40,
            MaxSequenceLength = 8,
            MaxCaptionLength = 3,
            DropoutRate = 0.0
        });

    [Fact]
    public void SigLIP2MaskTokens_ReplacesTheMaskedPatchEmbeddingsWithOneLearnedToken()
    {
        using var model = SmallSigLIP2();
        var tokens = Random(6, 4, 16);

        var masked = model.MaskTokens(tokens, new[] { 1, 3 });

        for (int d = 0; d < 16; d++)
        {
            Assert.Equal(tokens[0, d], masked[0, d]);
            Assert.Equal(tokens[2, d], masked[2, d]);
            Assert.Equal(masked[1, d], masked[3, d]);
        }
        Assert.Contains(Enumerable.Range(0, 16), d => masked[1, d] != tokens[1, d]);
    }

    [Fact]
    public void SigLIP2PredictMaskedPatches_PredictsAtTheMaskedPatchesOnly()
    {
        using var model = SmallSigLIP2();

        // 16-pixel patches on a 32 x 32 image: four patches.
        var predictions = model.PredictMaskedPatches(Random(7, 3, 32, 32), new[] { 0, 3 });

        Assert.Equal(2, predictions.Shape[0]);
        Assert.Equal(16, predictions.Shape[1]);
    }

    [Fact]
    public void SigLIP2CaptionDecoder_IsCausal_EarlierPositionsIgnoreLaterTokens()
    {
        using var model = SmallSigLIP2();
        var memory = Random(8, 4, 16);
        var sequence = Random(9, 3, 16);
        var changed = sequence.Clone();
        for (int d = 0; d < 16; d++) changed[2, d] = sequence[2, d] + 1.0;

        var before = model.DecodeCaption(sequence, memory);
        var after = model.DecodeCaption(changed, memory);

        for (int p = 0; p < 2; p++)
            for (int d = 0; d < 16; d++)
                Assert.Equal(before[p, d], after[p, d], Tolerance);
        Assert.Contains(Enumerable.Range(0, 16), d => Math.Abs(before[2, d] - after[2, d]) > 1e-12);
    }
}
