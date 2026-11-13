using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
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
        // Six backbone tokens (a 1 x 6 grid) make a two-level pyramid: 2 x 12 upsampled plus the 1 x 6 map, 30 tokens.
        Assert.All(selected, s => Assert.InRange(s, 0, 29));
        Assert.All(prediction.Boxes.ToArray(), v => Assert.True(v > 0 && v < 1));
    }

    [Fact]
    public void QueryDecoder_ClassAgnostic_ScoresQueriesByObjectness()
    {
        var head = new GroundingQueryDecoder<double>(Engine, new List<ILayer<double>>(),
            new GroundingQueryDecoderConfiguration(8, 2, 16, 0, 1, 10, 0, 1));

        var prediction = head.Forward(Random(2, 4, 5), null);

        // Four backbone tokens (2 x 2) make 4 x 4 + 2 x 2 + 1 x 1 = 21 multi-scale tokens, enough for the ten queries.
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
        attention.Offsets.SetParameters(new Vector<double>(attention.Offsets.ParameterCount));

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

        // The centre moves 10% of the way to the teacher's mean output.
        var teacherRows = model.TeacherPatchPredictions(image, masked);
        var centre = Enumerable.Range(0, 16).Select(j => 0.1 * (teacherRows[0, j] + teacherRows[1, j]) / 2).ToArray();
        Assert.Equal(Expected(centre), model.MaskedPredictionLoss(image, masked).ToArray()[0], 1e-9);
    }
}
