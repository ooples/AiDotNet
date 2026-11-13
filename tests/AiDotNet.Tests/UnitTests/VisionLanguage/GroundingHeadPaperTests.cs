using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.VisionLanguage;
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

        // Width 8: per coordinate four features, 2*pi*v / 10000^(2i/4) for i = 0, 1. The reference emits pos_y, pos_x,
        // pos_w, pos_h (the first two coordinates swapped), each torch.stack((sin, cos), dim=3).flatten(2), so sin and
        // cos of one frequency are adjacent: sin f0, cos f0, sin f1, cos f1.
        Assert.Equal(new[] { 2, 16 }, embedding.Shape.ToArray());
        var frequencies = new[] { 1.0, Math.Pow(10000, 0.5) };
        for (int k = 0; k < 2; k++)
            for (int block = 0; block < 4; block++)
            {
                int c = block < 2 ? 1 - block : block;
                for (int i = 0; i < 2; i++)
                {
                    double angle = 2 * Math.PI * boxes[k, c] / frequencies[i];
                    Assert.Equal(Math.Sin(angle), embedding[k, 4 * block + 2 * i], Tolerance);
                    Assert.Equal(Math.Cos(angle), embedding[k, 4 * block + 2 * i + 1], Tolerance);
                }
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
        // Six backbone tokens (a 1 x 6 grid) make four levels: 2 x 12 (transposed convolution), 1 x 6, then two
        // stride-2 convolutions (the map is too thin to pool), 1 x 3 and 1 x 2: 35 tokens.
        Assert.All(selected, s => Assert.InRange(s, 0, 34));
        Assert.Equal(2, prediction.LayerBoxes.Count);
        Assert.Equal(2, prediction.LayerLogits.Count);
        Assert.Same(prediction.Boxes, prediction.LayerBoxes[1]);
        Assert.Equal(new[] { 3, 4 }, prediction.EncoderBoxes.Shape.ToArray());
        Assert.Equal(new[] { 3, 3 }, prediction.EncoderLogits.Shape.ToArray());
        Assert.All(prediction.Boxes.ToArray(), v => Assert.True(v > 0 && v < 1));
    }

    [Fact]
    public void QueryDecoder_ClassAgnostic_ScoresQueriesByObjectness()
    {
        var head = new GroundingQueryDecoder<double>(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, 2, 16, 0, 1, 10, 0, 1));

        var prediction = head.Forward(Random(2, 4, 5), null);

        // Four backbone tokens (2 x 2) make 4 x 4 + 2 x 2 + 1 x 1 (pooled) + 1 x 1 (stride-2 convolution) = 22
        // multi-scale tokens, enough for the ten queries.
        Assert.Equal(new[] { 10, 4 }, prediction.Boxes.Shape.ToArray());
        Assert.Equal(new[] { 10, 1 }, prediction.Logits.Shape.ToArray());
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
        // The class token and nine patch tokens.
        var prediction = head.Forward(Random(3, 10, 8), query);

        Assert.Equal(new[] { 1, 6 }, query.Shape.ToArray());
        Assert.Equal(new[] { 9, 4 }, prediction.Boxes.Shape.ToArray());
        Assert.Equal(new[] { 9, 1 }, prediction.Logits.Shape.ToArray());
        Assert.Equal(objectness, prediction.Objectness is not null);
        if (prediction.Objectness is not null) Assert.Equal(new[] { 9 }, prediction.Objectness.Shape.ToArray());
    }

    // The decoder's characters: digits 0-9, then '.', ',', '[', ']', ';' and the end of the answer.
    private static double[] Characters(params int[] allowed)
    {
        var row = new double[16];
        foreach (int a in allowed) row[a] = 1;
        return row;
    }

    private const int Point = 10, Comma = 11, Open = 12, Close = 13, Semicolon = 14, End = 15;

    [Fact]
    public void CoordinateDecoder_SpellsWholeNumbers_DigitByDigit_InWellFormedBoxes()
    {
        // Ferret's spelling: the values 0-11 as whole numbers.
        var decoder = new CoordinateTokenDecoder<double>(Engine, new List<ILayer<double>>(),
            new CoordinateTokenDecoderConfiguration(8, 32, 4, Bins: 12, Divisor: 10, CoordinateSpelling.Integer));
        var digits = Enumerable.Range(0, 10).ToArray();
        int[] Box(params int[] values) => new[] { Open }.Concat(values.SelectMany((v, i) => (i > 0 ? new[] { Comma } : Array.Empty<int>())
            .Concat(v.ToString(System.Globalization.CultureInfo.InvariantCulture).Select(ch => ch - '0')))).Concat(new[] { Close }).ToArray();

        Assert.Equal(Characters(Open, End), decoder.Allowed(Array.Empty<int>(), 2).ToArray());
        Assert.Equal(Characters(digits), decoder.Allowed(new[] { Open }, 2).ToArray());
        // "1" is a value and also starts "10" and "11"; "0" has no longer spelling.
        Assert.Equal(Characters(0, 1, Comma), decoder.Allowed(new[] { Open, 1 }, 2).ToArray());
        Assert.Equal(Characters(Comma), decoder.Allowed(new[] { Open, 0 }, 2).ToArray());
        Assert.Equal(Characters(Close), decoder.Allowed(new[] { Open, 1, Comma, 2, Comma, 3, Comma, 4 }, 2).ToArray());
        var one = Box(1, 2, 3, 11);
        Assert.Equal(Characters(Semicolon, End), decoder.Allowed(one, 2).ToArray());
        Assert.Equal(Characters(Open), decoder.Allowed(one.Concat(new[] { Semicolon }).ToArray(), 2).ToArray());
        Assert.Equal(Characters(End), decoder.Allowed(one.Concat(new[] { Semicolon }).Concat(Box(0, 0, 10, 10)).ToArray(), 2).ToArray());
    }

    [Fact]
    public void CoordinateDecoder_SpellsDecimalFractions_AsShikraDoes()
    {
        // Shikra's spelling with one decimal place: 0.0, 0.1, ..., 1.0.
        var decoder = new CoordinateTokenDecoder<double>(Engine, new List<ILayer<double>>(),
            new CoordinateTokenDecoderConfiguration(8, 32, 4, Bins: 11, Divisor: 10, CoordinateSpelling.DecimalFraction));

        Assert.Equal(Characters(0, 1), decoder.Allowed(new[] { Open }, 1).ToArray());
        Assert.Equal(Characters(Point), decoder.Allowed(new[] { Open, 1 }, 1).ToArray());
        Assert.Equal(Characters(0), decoder.Allowed(new[] { Open, 1, Point }, 1).ToArray());
        Assert.Equal(Characters(Enumerable.Range(0, 10).ToArray()), decoder.Allowed(new[] { Open, 0, Point }, 1).ToArray());
        Assert.Equal(Characters(Comma), decoder.Allowed(new[] { Open, 0, Point, 7 }, 1).ToArray());
    }

    [Fact]
    public void CoordinateDecoder_WritesBoxesAsText_ScoredByTheirProbability()
    {
        var decoder = new CoordinateTokenDecoder<double>(Engine, new List<ILayer<double>>(),
            new CoordinateTokenDecoderConfiguration(8, 32, 4, Bins: 11, Divisor: 10, CoordinateSpelling.DecimalFraction));

        // A stand-in language model that passes its input through.
        var (boxes, scores) = decoder.Ground(Random(4, 3, 8), new[] { 1, 2 }, rows => rows, maxBoxes: 3);

        int count = scores.Length;
        Assert.InRange(count, 0, 3);
        Assert.Equal(new[] { count, 4 }, boxes.Shape.ToArray());
        // Every coordinate is a tenth: 0, 0.1, ..., 1.
        Assert.All(boxes.ToArray(), v => Assert.Equal(0.0, v * 10 - Math.Round(v * 10), 1e-9));
        Assert.All(scores.ToArray(), s => Assert.InRange(s, 0.0, 1.0));
    }

    [Fact]
    public void Groma_RegionFeatures_AreTorchvisionAlignedRoIAlignMeans()
    {
        var grounder = new GromaRegionGrounder<double>(Engine, new List<ILayer<double>>(),
            new GromaRegionGrounderConfiguration(8, 8, 2, 1, 3, 32, 4));
        // A 2 x 2 map v(x, y) = 1 + x + 2y on pixel centres: [[1, 2], [3, 4]].
        var tokens = Matrix(4, 1, 1, 2, 3, 4);
        var boxes = Matrix(2, 4,
            0.0, 0.0, 1.0, 1.0,     // the whole map: symmetric samples, the mean 2.5
            0.0, 0.0, 0.5, 1.0);    // the left half

        var features = grounder.AlignedRegions(tokens, 2, 2, boxes).ToArray();

        // Left half, aligned: x spans pixel coordinates [-0.5, 0.5] in 14 samples, those below 0 clamped to 0, so the
        // mean x is (1 + 3 + ... + 13) / 28 / 14 = 0.125; y spans [-0.5, 1.5], clamped to [0, 1], mean 0.5.
        // v = 1 + 0.125 + 2 * 0.5 = 2.125. (Averaging the tokens whose centre lies in the box would give 2.)
        Assert.Equal(2.5, features[0], 1e-9);
        Assert.Equal(2.125, features[1], 1e-9);
    }

    private sealed class HeadProbe : GroundingHeadModule<double>
    {
        public HeadProbe() : base(AiDotNetEngine.Current, new List<ILayer<double>>()) { }

        public DeformableAttentionLayers<double> Deformable(int width, int heads, int levels, int points)
            => DeformableAttention(width, heads, levels, points);

        public FeaturePyramidLayers<double> PyramidOf(int width, int levels) => PyramidLayers(width, levels);
    }

    [Fact]
    public void Pyramid_IsViTDetsSimpleFeaturePyramid_UpsampledThenPooledLevels()
    {
        var probe = new HeadProbe();
        var levels = probe.Pyramid(probe.PyramidOf(4, 4), Random(10, 16, 3));

        Assert.Equal(new[] { (8, 8), (4, 4), (2, 2), (1, 1) }, levels.Shapes);
        Assert.Equal(new[] { 0, 64, 80, 84 }, levels.Starts);
        Assert.Equal(new[] { 85, 4 }, levels.Tokens.Shape.ToArray());
        Assert.Equal(new[] { 85, 4 }, levels.LevelEmbedding.Shape.ToArray());
        // Level 2 (2 x 2): token 3's centre is (0.75, 0.75).
        Assert.Equal(0.75, levels.Centres[83, 0], Tolerance);
        Assert.Equal(0.75, levels.Centres[83, 1], Tolerance);
    }

    [Fact]
    public void Deform_WithZeroOffsets_SamplesTheValueAtTheReference()
    {
        var probe = new HeadProbe();
        var attention = probe.Deformable(4, 2, 1, 3);
        var tokens = Random(11, 4, 4);
        var levels = new FeatureLevels<double>(tokens, new Tensor<double>(new[] { 4, 4 }),
            GroundingHeadModule<double>.GridCentres(Engine, 2, 2), new[] { (2, 2) }, new[] { 0 });
        var query = Random(12, 1, 4);
        probe.Deform(attention, query, Matrix(1, 2, 0.75, 0.25), levels);
        attention.Offsets.SetParameters(new Vector<double>(attention.Offsets.GetParameters().Length));

        // Token 1's centre (0.75, 0.25) lands on its pixel exactly, and every point samples there, so the weights
        // (summing to 1 per head) leave that token's projected value: Output(Value(tokens)[1]).
        var atPoint = probe.Deform(attention, query, Matrix(1, 2, 0.75, 0.25), levels);
        var atBox = probe.Deform(attention, query, Matrix(1, 4, 0.75, 0.25, 0.3, 0.6), levels);
        var value = attention.Value.Forward(tokens);
        var expected = attention.Output.Forward(Engine.TensorSlice(value, new[] { 1, 0 }, new[] { 1, 4 }));

        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(expected[0, d], atPoint[0, d], 1e-9);
            Assert.Equal(expected[0, d], atBox[0, d], 1e-9);
        }
    }

    [Fact]
    public void OwlHead_MergesTheClassTokenIntoThePatches_ThenNormalizes()
    {
        var head = new OwlDetectionHead<double>(Engine, new List<ILayer<double>>(), 2,
            new OwlDetectionHeadConfiguration(6, 8, 1, 2, 32, 4, false));
        // Class token (2, 2) first, then the patches (1, 3) and (3, 1): products (2, 6) and (6, 2), each normalized
        // to mean 0 and variance 1 (variance 4, epsilon 1e-5).
        var merged = head.MergeClassToken(Matrix(3, 2, 2, 2, 1, 3, 3, 1));

        double unit = 2 / Math.Sqrt(4 + 1e-5);
        Assert.Equal(new[] { 2, 2 }, merged.Shape.ToArray());
        Assert.Equal(new[] { -unit, unit, unit, -unit }, merged.ToArray(), new ToleranceComparer(1e-9));
    }

    private sealed class ToleranceComparer : IEqualityComparer<double>
    {
        private readonly double _tolerance;
        public ToleranceComparer(double tolerance) => _tolerance = tolerance;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _tolerance;
        public int GetHashCode(double obj) => 0;
    }

    [Fact]
    public void SetLoss_PairwiseGIoU_MatchesTheReferenceGeneralizedIoU()
    {
        var loss = new GroundingSetLoss<double>(Engine);
        var a = new[] { new[] { 0.5, 0.5, 0.4, 0.4 }, new[] { 0.2, 0.3, 0.1, 0.2 } };
        var b = new[] { new[] { 0.6, 0.5, 0.2, 0.4 }, new[] { 0.9, 0.9, 0.1, 0.1 }, new[] { 0.5, 0.5, 0.4, 0.4 } };

        var giou = loss.PairwiseGeneralizedIoU(Matrix(2, 4, a.SelectMany(v => v).ToArray()), Matrix(3, 4, b.SelectMany(v => v).ToArray()));

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(GroundingDinoLoss<double>.GeneralizedIoU(a[i], b[j]), giou[i, j], 1e-12);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SetLoss_MatchesTheExactQuery_AndTakesFocalL1AndGIoUPerBox(bool swapped)
    {
        var loss = new GroundingSetLoss<double>(Engine);
        var exact = new[] { 0.5, 0.5, 0.2, 0.2 };
        var off = new[] { 0.2, 0.2, 0.1, 0.1 };
        var boxes = Matrix(2, 4, (swapped ? exact.Concat(off) : off.Concat(exact)).ToArray());

        var terms = loss.Compute(boxes, new Tensor<double>(new[] { 2, 1 }), Matrix(1, 4, exact), Matrix(1, 1, 1));

        // The exact query is matched, so L1 and 1 - GIoU are 0. With p = 0.5 everywhere the focal loss is
        // alpha (1 - p)^2 ln 2 for the matched query plus (1 - alpha) p^2 ln 2 for the other: 0.25 ln 2, times weight 2.
        Assert.Equal(new[] { swapped ? 0 : 1 }, terms.QueryOfTarget);
        Assert.Equal(0.25 * Math.Log(2), terms.Classification.ToArray()[0], 1e-12);
        Assert.Equal(0.0, terms.Box.ToArray()[0], 1e-12);
        Assert.Equal(0.0, terms.GeneralizedIoU.ToArray()[0], 1e-12);
        Assert.Equal(0.5 * Math.Log(2), terms.Total.ToArray()[0], 1e-12);
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

    // A three-layer decoder's outputs on one image: distinct boxes and logits per layer and for the encoder.
    private static GroundingQueryPrediction<double> ThreeLayerPrediction()
    {
        Tensor<double> Boxes(int seed) => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.TensorSigmoid(Random(seed, 4, 4)), 0.5), 0.1);
        var layerBoxes = new[] { Boxes(20), Boxes(21), Boxes(22) };
        var layerLogits = new[] { Random(23, 4, 3), Random(24, 4, 3), Random(25, 4, 3) };
        return new GroundingQueryPrediction<double>(layerBoxes[2], layerLogits[2], new Tensor<int>(new[] { 4 }),
            layerBoxes, layerLogits, Boxes(26), Random(27, 4, 3));
    }

    private static readonly Tensor<double> TargetBoxes = Matrix(2, 4, 0.3, 0.3, 0.2, 0.2, 0.7, 0.6, 0.3, 0.4);
    private static readonly Tensor<double> TargetTokens = Matrix(2, 3, 1, 1, 0, 0, 0, 1);

    private static double SetLossOf(Tensor<double> boxes, Tensor<double> logits)
        => new GroundingSetLoss<double>(Engine).Compute(boxes, logits, TargetBoxes, TargetTokens).Total.ToArray()[0];

    [Fact]
    public void DetectorLoss_IncludesTheLastDecoderLayersSetLoss()
    {
        var p = ThreeLayerPrediction();
        var loss = new GroundingSetLoss<double>(Engine).Compute(p, TargetBoxes, TargetTokens);

        double final = SetLossOf(p.Boxes, p.Logits);
        Assert.True(final > 0);
        Assert.Equal(final, loss.Final.Total.ToArray()[0], 1e-12);
        Assert.Equal(final, loss.Total.ToArray()[0] - SetLossOf(p.LayerBoxes[0], p.LayerLogits[0])
            - SetLossOf(p.LayerBoxes[1], p.LayerLogits[1]) - SetLossOf(p.EncoderBoxes, p.EncoderLogits), 1e-9);
    }

    [Fact]
    public void DetectorLoss_IncludesEveryEarlierDecoderLayersAuxiliarySetLoss()
    {
        var p = ThreeLayerPrediction();
        var loss = new GroundingSetLoss<double>(Engine).Compute(p, TargetBoxes, TargetTokens);

        double auxiliary = SetLossOf(p.LayerBoxes[0], p.LayerLogits[0]) + SetLossOf(p.LayerBoxes[1], p.LayerLogits[1]);
        Assert.Equal(2, loss.Auxiliary.Count);
        Assert.True(auxiliary > 0);
        Assert.Equal(auxiliary, loss.Total.ToArray()[0] - SetLossOf(p.Boxes, p.Logits) - SetLossOf(p.EncoderBoxes, p.EncoderLogits), 1e-9);
    }

    [Fact]
    public void DetectorLoss_IncludesTheEncoderProposalsSetLoss()
    {
        var p = ThreeLayerPrediction();
        var loss = new GroundingSetLoss<double>(Engine).Compute(p, TargetBoxes, TargetTokens);

        double encoder = SetLossOf(p.EncoderBoxes, p.EncoderLogits);
        Assert.True(encoder > 0);
        Assert.Equal(encoder, loss.Total.ToArray()[0] - SetLossOf(p.Boxes, p.Logits)
            - SetLossOf(p.LayerBoxes[0], p.LayerLogits[0]) - SetLossOf(p.LayerBoxes[1], p.LayerLogits[1]), 1e-9);
    }

    [Fact]
    public void Pyramid_NormalizesEachLevelsChannelsOverTheMap_AsDinosGroupNorm()
    {
        var probe = new HeadProbe();
        // Width 4 splits into 4 groups of one channel, so each channel is normalized over its level's whole map
        // (a layer norm would normalize each token over its channels instead).
        var levels = probe.Pyramid(probe.PyramidOf(4, 4), Random(28, 4, 3));

        Assert.Equal(new[] { (4, 4), (2, 2), (1, 1), (1, 1) }, levels.Shapes);
        for (int level = 0; level < 2; level++)
        {
            int start = levels.Starts[level], count = levels.Shapes[level].H * levels.Shapes[level].W;
            for (int d = 0; d < 4; d++)
            {
                var values = Enumerable.Range(start, count).Select(t => levels.Tokens[t, d]).ToArray();
                double mean = values.Average();
                double variance = values.Select(v => (v - mean) * (v - mean)).Average();
                Assert.Equal(0.0, mean, 1e-9);
                Assert.InRange(variance, 0.99, 1.0);
            }
        }
    }

    [Fact]
    public void SigLIP2CaptioningLoss_TeacherForcedOnTheGreedyCaption_IsTheGreedyStepsCrossEntropy()
    {
        using var model = SmallSigLIP2();
        var image = Random(13, 3, 32, 32);
        var stepLogits = model.GenerateCaption(image);
        int steps = stepLogits.Shape[0], vocabulary = stepLogits.Shape[1];
        var greedy = new int[steps];
        double expected = 0;
        for (int step = 0; step < steps; step++)
        {
            var row = Enumerable.Range(0, vocabulary).Select(v => stepLogits[step, v]).ToArray();
            greedy[step] = Array.IndexOf(row, row.Max());
            double max = row.Max();
            double logSum = max + Math.Log(row.Sum(v => Math.Exp(v - max)));
            expected += logSum - row[greedy[step]];
        }

        // Teacher forcing on the tokens greedy decoding chose feeds the decoder exactly what decoding fed it.
        var loss = model.CaptioningLoss(image, greedy);

        Assert.Equal(expected / steps, loss.ToArray()[0], 1e-9);
    }

    [Fact]
    public void SigLIP2CaptioningLoss_ParallelPrediction_IgnoresTheReferenceTokens()
    {
        using var model = SmallSigLIP2();
        var memory = Random(14, 4, 16);
        double Loss(int[] tokens, bool parallel) => model.CaptioningLossFromMemory(memory, tokens, parallel).ToArray()[0];

        // The last position's term (the difference between its two targets) cannot depend on the first token when
        // every decoder input is the start token; under teacher forcing it reads the first token.
        double parallelA = Loss(new[] { 5, 7, 9 }, true) - Loss(new[] { 5, 7, 11 }, true);
        double parallelB = Loss(new[] { 8, 7, 9 }, true) - Loss(new[] { 8, 7, 11 }, true);
        double forcedA = Loss(new[] { 5, 7, 9 }, false) - Loss(new[] { 5, 7, 11 }, false);
        double forcedB = Loss(new[] { 8, 7, 9 }, false) - Loss(new[] { 8, 7, 11 }, false);

        Assert.Equal(parallelA, parallelB, 1e-9);
        Assert.NotEqual(forcedA, forcedB, 6);
    }

    [Fact]
    public void EmaTeacher_RunsOnItsOwnParameters_AndFollowsTheStudentWithMomentum()
    {
        var layer = new DenseLayer<double>(2, (IActivationFunction<double>?)null);
        var x = Random(15, 1, 3);
        var before = layer.Forward(x);
        var teacher = new EmaTeacher<double>(Engine, new List<ILayer<double>> { layer }, 0.5);
        teacher.Update();
        var original = layer.GetParameters().ToArray();
        var moved = original.Select(v => v + 1).ToArray();
        layer.SetParameters(new Vector<double>(moved));

        var teacherOutput = teacher.Run(() => layer.Forward(x));

        Assert.Equal(before.ToArray(), teacherOutput.ToArray(), new ToleranceComparer(1e-12));
        Assert.Equal(moved, layer.GetParameters().ToArray());
        teacher.Update();
        Assert.Equal(original.Select(v => v + 0.5).ToArray(), teacher.Parameters[0].ToArray(), new ToleranceComparer(1e-12));
    }

    [Fact]
    public void SigLIP2MaskedPredictionLoss_IsTheCentredTeacherCrossEntropy()
    {
        using var model = SmallSigLIP2();
        var image = Random(16, 3, 32, 32);
        var masked = new[] { 0, 2 };

        double Expected(double[] centre)
        {
            var s = model.PredictMaskedPatches(image, masked);
            var t = model.TeacherPatchPredictions(image, masked);
            int n = s.Shape[0], f = s.Shape[1];
            double total = 0;
            for (int i = 0; i < n; i++)
            {
                var teacher = Enumerable.Range(0, f).Select(j => (t[i, j] - centre[j]) / 0.04).ToArray();
                var student = Enumerable.Range(0, f).Select(j => s[i, j] / 0.1).ToArray();
                double tMax = teacher.Max(), sMax = student.Max();
                double tSum = teacher.Sum(v => Math.Exp(v - tMax));
                double sLog = sMax + Math.Log(student.Sum(v => Math.Exp(v - sMax)));
                for (int j = 0; j < f; j++) total -= Math.Exp(teacher[j] - tMax) / tSum * (student[j] - sLog);
            }
            return total / n;
        }

        double first = Expected(new double[16]);
        Assert.Equal(first, model.MaskedPredictionLoss(image, masked).ToArray()[0], 1e-9);

        // The centre moves 10% of the way to the teacher's mean output over every patch of the image (iBOT's patch
        // centre is teacher_patch.mean(1), over all patches, not only the masked ones).
        var teacherRows = model.TeacherPatchPredictions(image, null);
        Assert.Equal(4, teacherRows.Shape[0]);
        var centre = Enumerable.Range(0, 16).Select(j => 0.1 * Enumerable.Range(0, 4).Average(r => teacherRows[r, j])).ToArray();
        var stored = model.MaskedPredictionCenter;
        Assert.NotNull(stored);
        if (stored is not null) Assert.Equal(centre, stored.ToArray(), new ToleranceComparer(1e-12));
        Assert.Equal(Expected(centre), model.MaskedPredictionLoss(image, masked).ToArray()[0], 1e-9);
    }

    // Rewinds the arena and floods its scratch tier with poisoned tensors of the given lengths, so anything kept in a
    // recycled buffer is overwritten.
    private static void ResetAndClobber(TensorArena arena, params int[] lengths)
    {
        arena.Reset();
        foreach (int length in lengths)
            for (int rep = 0; rep < 256; rep++)
            {
                var t = TensorAllocator.Rent<double>(new[] { length });
                var span = t.Data.Span;
                for (int i = 0; i < span.Length; i++) span[i] = 123456.0;
            }
    }

    [Fact]
    public void SigLIP2MaskedPredictionCentre_SurvivesTheArenaRecyclingItsBuffers()
    {
        using var model = SmallSigLIP2();
        var image = Random(17, 3, 32, 32);
        using var arena = TensorArena.Create();
        model.MaskedPredictionLoss(image, new[] { 1, 3 });
        var stored = model.MaskedPredictionCenter;
        Assert.NotNull(stored);
        var before = stored?.ToArray() ?? Array.Empty<double>();

        ResetAndClobber(arena, 16, 4 * 16);

        Assert.Equal(before, model.MaskedPredictionCenter?.ToArray() ?? Array.Empty<double>());
    }

    [Fact]
    public void EmaTeacher_Parameters_SurviveTheArenaRecyclingItsBuffers()
    {
        using var arena = TensorArena.Create();
        var layer = new DenseLayer<double>(2, (IActivationFunction<double>?)null);
        layer.Forward(Random(18, 1, 3));
        var teacher = new EmaTeacher<double>(Engine, new List<ILayer<double>> { layer }, 0.5);
        teacher.Update();
        layer.SetParameters(new Vector<double>(layer.GetParameters().ToArray().Select(v => v + 1).ToArray()));
        teacher.Update();
        var before = teacher.Parameters[0].ToArray();

        ResetAndClobber(arena, before.Length);

        Assert.Equal(before, teacher.Parameters[0].ToArray());
    }

    private static GroundingQueryDecoder<double> Decoder(int enhancerLayers, int decoderLayers, int queries = 3, int heads = 2)
        => new(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, heads, 16, enhancerLayers, decoderLayers, queries, 32, 4));

    private static double GradientMagnitude(Dictionary<Tensor<double>, Tensor<double>> gradients, IEnumerable<Tensor<double>> parameters)
        => parameters.Sum(p => gradients.TryGetValue(p, out var g) ? g.ToArray().Sum(Math.Abs) : 0.0);

    [Fact]
    public void QueryDecoder_StartsFromDetachedProposals_AsDinosRefpointEmbedDetach()
    {
        var head = Decoder(1, 2);
        var image = Random(30, 6, 5);
        var ids = new[] { 3, 7, 11 };
        head.Forward(image, ids);
        var encoderBoxHead = TapeTrainingStep<double>.CollectParameters(head.EncoderBoxHead, -1).ToList();
        var boxHead = TapeTrainingStep<double>.CollectParameters(head.BoxHead, -1).ToList();

        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var prediction = head.Forward(image, ids);
            var loss = Engine.ReduceSum(prediction.Boxes, new[] { 0, 1 }, false);
            gradients = tape.ComputeGradients(loss, sources: encoderBoxHead.Concat(boxHead).ToList());
        }

        // The decoder's boxes refine the proposals without differentiating them; the shared box head is trained.
        Assert.Equal(0.0, GradientMagnitude(gradients, encoderBoxHead));
        Assert.True(GradientMagnitude(gradients, boxHead) > 0);
    }

    [Fact]
    public void QueryDecoder_LaterLayersReadDetachedBoxes_SoTheirLogitsDoNotTrainTheBoxHead()
    {
        var head = Decoder(1, 2);
        var image = Random(31, 6, 5);
        var ids = new[] { 3, 7, 11 };
        head.Forward(image, ids);
        var boxHead = TapeTrainingStep<double>.CollectParameters(head.BoxHead, -1).ToList();
        var referenceHead = TapeTrainingStep<double>.CollectParameters(head.ReferenceHead, -1).ToList();

        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var prediction = head.Forward(image, ids);
            var loss = Engine.ReduceSum(prediction.LayerLogits[1], new[] { 0, 1 }, false);
            gradients = tape.ComputeGradients(loss, sources: boxHead.Concat(referenceHead).ToList());
        }

        // The second layer's queries see the first layer's box only through reference_points.detach().
        Assert.Equal(0.0, GradientMagnitude(gradients, boxHead));
        Assert.True(GradientMagnitude(gradients, referenceHead) > 0);
    }

    [Fact]
    public void QueryDecoder_TextCrossAttention_ReadsTheQueryWithItsPosition()
    {
        var head = Decoder(0, 1);
        var image = Random(32, 6, 5);
        var ids = new[] { 3, 7, 11 };
        head.Forward(image, ids);
        var layer = head.DecoderLayers[0];
        void Zero(ILayer<double> l) => l.SetParameters(new Vector<double>(l.GetParameters().Length));
        // Silence the self-attention, the image attention and the feed-forward: the position then reaches the logits
        // only through the text cross-attention, whose query is with_pos_embed(tgt, tgt_query_pos).
        Zero(layer.SelfAttention);
        Zero(layer.ImageAttention.Output);
        Zero(layer.FeedForwardDown);
        var before = head.Forward(image, ids).Logits.ToArray();
        var positionLayer = head.ReferenceHead[1];
        positionLayer.SetParameters(new Vector<double>(positionLayer.GetParameters().ToArray().Select(v => v + 0.5).ToArray()));

        var after = head.Forward(image, ids).Logits.ToArray();

        Assert.Contains(Enumerable.Range(0, before.Length), i => Math.Abs(before[i] - after[i]) > 1e-6);
    }

    private static FeatureLevels<double> SingleLevel(int seed, int h, int w, int width)
        => new(Random(seed, h * w, width), new Tensor<double>(new[] { h * w, width }),
            GroundingHeadModule<double>.GridCentres(Engine, h, w), new[] { (h, w) }, new[] { 0 });

    [Fact]
    public void QueryDecoder_EnhancerLayers_ArePostNorm_AsDeformableDetrAndGroundingDino()
    {
        var head = Decoder(1, 1);
        var (image, text) = head.Enhance(SingleLevel(33, 2, 2, 8), head.EmbedText(new[] { 3, 7, 11 }));

        // Each side ends with norm2(src + ffn(src)), a freshly initialized layer norm: every row has mean 0, variance ~1.
        Assert.NotNull(text);
        foreach (var rows in new[] { image, text ?? image })
            for (int r = 0; r < rows.Shape[0]; r++)
            {
                var row = Enumerable.Range(0, 8).Select(d => rows[r, d]).ToArray();
                double mean = row.Average();
                Assert.Equal(0.0, mean, 1e-9);
                Assert.InRange(row.Select(v => (v - mean) * (v - mean)).Average(), 0.99, 1.0);
            }
        Assert.Equal(16 / 2, head.EnhancerLayers[0].Text?.FeedForwardUp.GetOutputShape()[0] ?? 0);
    }

    [Fact]
    public void QueryDecoder_Fusion_IsGroundingDinosBiMultiHeadAttention_WithLayerScale()
    {
        // Feed-forward 16: the fusion embeds in 8 features over 4 / 2 = 2 heads.
        var head = Decoder(1, 1, heads: 4);
        var fusion = head.EnhancerLayers[0].Fusion;
        Assert.NotNull(fusion);
        if (fusion is null) return;
        var image = Random(34, 5, 8);
        var text = Random(35, 3, 8);

        var (updatedImage, updatedText) = head.BiAttend(fusion, image, text);

        int h = fusion.Heads, dh = fusion.Width / h;
        Assert.Equal(2, h);
        var one = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 }));
        var imageScale = fusion.ImageScale.Forward(one);
        var textScale = fusion.TextScale.Forward(one);
        Assert.All(imageScale.ToArray().Concat(textScale.ToArray()), s => Assert.Equal(1e-4, s, 15));

        var v = fusion.ImageNorm.Forward(image);
        var l = fusion.TextNorm.Forward(text);
        var q = fusion.ImageQuery.Forward(v);
        var k = fusion.TextKey.Forward(l);
        var imageValues = fusion.ImageValue.Forward(v);
        var textValues = fusion.TextValue.Forward(l);
        var toText = new double[5, 8];
        var toImage = new double[3, 8];
        for (int head2 = 0; head2 < h; head2++)
        {
            var logits = new double[5, 3];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++)
                    for (int f = 0; f < dh; f++)
                        logits[i, j] += q[i, head2 * dh + f] / Math.Sqrt(dh) * k[j, head2 * dh + f];
            // The image update: softmax over the text; the text update: the same logits softmaxed over the image.
            for (int i = 0; i < 5; i++)
            {
                double max = Enumerable.Range(0, 3).Max(j => logits[i, j]);
                double sum = Enumerable.Range(0, 3).Sum(j => Math.Exp(logits[i, j] - max));
                for (int j = 0; j < 3; j++)
                    for (int f = 0; f < dh; f++)
                        toText[i, head2 * dh + f] += Math.Exp(logits[i, j] - max) / sum * textValues[j, head2 * dh + f];
            }
            for (int j = 0; j < 3; j++)
            {
                double max = Enumerable.Range(0, 5).Max(i => logits[i, j]);
                double sum = Enumerable.Range(0, 5).Sum(i => Math.Exp(logits[i, j] - max));
                for (int i = 0; i < 5; i++)
                    for (int f = 0; f < dh; f++)
                        toImage[j, head2 * dh + f] += Math.Exp(logits[i, j] - max) / sum * imageValues[i, head2 * dh + f];
            }
        }
        var deltaImage = fusion.ImageOutput.Forward(new Tensor<double>(new[] { 5, 8 }, new Vector<double>(toText.Cast<double>().ToArray())));
        var deltaText = fusion.TextOutput.Forward(new Tensor<double>(new[] { 3, 8 }, new Vector<double>(toImage.Cast<double>().ToArray())));
        for (int d = 0; d < 8; d++)
        {
            for (int i = 0; i < 5; i++)
                Assert.Equal(image[i, d] + imageScale[0, d] * deltaImage[i, d], updatedImage[i, d], 1e-12);
            for (int j = 0; j < 3; j++)
                Assert.Equal(text[j, d] + textScale[0, d] * deltaText[j, d], updatedText[j, d], 1e-12);
        }
    }

    [Fact]
    public void QueryDecoder_ImagePosition_IsGroundingDinosSineHW_Temperature20()
    {
        var head = Decoder(1, 1);

        var position = head.ImagePosition(SingleLevel(36, 2, 3, 8));

        // PositionEmbeddingSineHW(normalize=True, temperature 20): y = (i + 1) / h and x = (j + 1) / w, 4 features
        // each (pos_y first), sin and cos of 2*pi*v / 20^(2i/4) interleaved.
        Assert.Equal(new[] { 6, 8 }, position.Shape.ToArray());
        var frequencies = new[] { 1.0, Math.Pow(20, 0.5) };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                int token = i * 3 + j;
                var axes = new[] { (i + 1) / 2.0, (j + 1) / 3.0 };
                for (int axis = 0; axis < 2; axis++)
                    for (int f = 0; f < 2; f++)
                    {
                        double angle = 2 * Math.PI * axes[axis] / frequencies[f];
                        Assert.Equal(Math.Sin(angle), position[token, 4 * axis + 2 * f], Tolerance);
                        Assert.Equal(Math.Cos(angle), position[token, 4 * axis + 2 * f + 1], Tolerance);
                    }
            }
    }

    [Fact]
    public void QueryDecoder_InvalidProposals_HaveZeroMemoryAndInfiniteLogits_AsGenEncoderOutputProposals()
    {
        // 26 tokens in one row make 2 x 52, 1 x 26, 1 x 13 and 1 x 7 levels: 150 tokens, every one a query.
        var head = Decoder(0, 1, queries: 150);

        var prediction = head.Forward(Random(37, 26, 5), new[] { 3, 7 });

        var selected = prediction.Selected.ToArray();
        Assert.Equal(150, selected.Length);
        for (int query = 0; query < selected.Length; query++)
        {
            int token = selected[query];
            // The finest level's first and last columns sit at x = 0.5 / 52 and 51.5 / 52, outside (0.01, 0.99).
            bool invalid = token < 104 && (token % 52 == 0 || token % 52 == 51);
            if (invalid)
                for (int c = 0; c < 4; c++) Assert.Equal(1.0, prediction.EncoderBoxes[query, c]);
            else
                Assert.True(prediction.EncoderBoxes[query, 0] < 1.0);
        }
    }

    [Fact]
    public void SetLoss_HalfWithSaturatedLogits_MatchesOnAFiniteCost()
    {
        // sigmoid(-30) underflows Half to 0, and 1e-8 rounds to 0 in Half, so the reference epsilon alone would leave
        // log(0) in the focal matching cost and the assignment could not run.
        var ops = MathHelper.GetNumericOperations<Half>();
        Tensor<Half> H(int rows, int columns, params double[] values)
            => new(new[] { rows, columns }, new Vector<Half>(values.Select(ops.FromDouble).ToArray()));
        var loss = new GroundingSetLoss<Half>(Engine);

        var terms = loss.Compute(H(2, 4, 0.2, 0.2, 0.1, 0.1, 0.5, 0.5, 0.2, 0.2), H(2, 2, -30, -30, -30, -30),
            H(1, 4, 0.5, 0.5, 0.2, 0.2), H(1, 2, 1, 0));

        Assert.Equal(new[] { 1 }, terms.QueryOfTarget);
        double total = ops.ToDouble(terms.Total.ToArray()[0]);
        Assert.False(double.IsNaN(total) || double.IsInfinity(total));
        Assert.Equal(1e-8, GroundingSetLoss<double>.CostEpsilon(MathHelper.GetNumericOperations<double>()));
    }

    [Fact]
    public void Ranked_TiedScores_KeepIndexOrder_AsTorchvisionsStableSort()
    {
        // Forty disjoint boxes with one score come back in index order.
        const int count = 40;
        var values = new List<double>();
        for (int b = 0; b < count; b++) values.AddRange(new[] { 0.02 * b, 0.0, 0.02 * b + 0.01, 0.01 });
        var scores = new Tensor<double>(new[] { count }, new Vector<double>(Enumerable.Repeat(0.5, count).ToArray()));

        var rows = GroundingDetections<double>.Ranked(Engine, Matrix(count, 4, values.ToArray()), scores, 0.3, count, 0.5).ToArray();

        Assert.Equal(count * 5, rows.Length);
        for (int b = 0; b < count; b++) Assert.Equal(0.02 * b, rows[5 * b], 1e-12);

        // Two overlapping boxes with one score: the earlier is kept.
        var overlapping = Matrix(2, 4, 0.0, 0.0, 1.0, 0.98, 0.0, 0.0, 1.0, 1.0);
        var tied = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.7, 0.7 }));
        Assert.Equal(new[] { 0.0, 0.0, 1.0, 0.98, 0.7 },
            GroundingDetections<double>.Ranked(Engine, overlapping, tied, 0.3, 2, 0.5).ToArray());
    }
}
