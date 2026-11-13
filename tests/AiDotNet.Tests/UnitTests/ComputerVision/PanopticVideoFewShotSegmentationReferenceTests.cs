using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Hand-derived references for the paper rewrites of kMaX-DeepLab (multi-scale k-means decoder, set-prediction
/// loss and panoptic merge), ODISE (Mask2Former loss and merge, CLIP image fusion with the learned diffusion scale),
/// CUPS (pseudo-label fusion, thing/stuff split, Cascade Mask R-CNN targets and paste, Panoptic FPN merge), DEVA
/// (memory reading, soft aggregation, no input residual), UniVS (prompt memory) and UniverSeg (CrossBlock averaged
/// over the support set, empty support rejected).
/// </summary>
public class PanopticVideoFewShotSegmentationReferenceTests
{
    private const double Tolerance = 1e-6;

    private static Tensor<double> T(int[] shape, params double[] values)
        => new Tensor<double>(shape, new Vector<double>(values));

    private static void AssertValues(Tensor<double> actual, params double[] expected)
    {
        var values = actual.ToArray();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], values[i], 6);
    }

    [Fact]
    public void KMeansDecoder_ReadsTheScalesCoarseToFineWithArgmaxAssignments()
    {
        // D = 2, N = 2, initial centers c0 = (1, 0), c1 = (0, 1). Two updates over two scales: update 0 reads the
        // coarse scale (one pixel (3, 4) -> (0.6, 0.8)), update 1 the fine scale (pixels (1, 0) and (0, 2) -> (0, 1)).
        var coarse = T(new[] { 1, 2, 1 }, 3, 4);
        var fine = T(new[] { 1, 2, 2 }, 1, 0, 0, 2);

        var (centers, masks) = MaskClassificationOps.KMeansDecoder(
            new[] { coarse, fine }, 2, T(new[] { 2, 2 }, 1, 0, 0, 1), fine);

        // Update 0: affinities [0.6, 0.8] -> c1 += (0.6, 0.8) = (0.6, 1.8).
        // Update 1: (1, 0) has affinities [1, 0.6] -> c0 += (1, 0) = (2, 0); (0, 1) has [0, 1.8] -> c1 = (0.6, 2.8).
        AssertValues(centers, 2, 0, 0.6, 2.8);
        // Mask logits on the normalized finest pixels (1, 0), (0, 1): c0 -> [2, 0], c1 -> [0.6, 2.8].
        AssertValues(masks, 2, 0, 0.6, 2.8);
    }

    [Fact]
    public void FeaturePyramid_TopDownAddsTheUpsampledCoarserLateralBeforeTheOutputConvolution()
    {
        var engine = AiDotNetEngine.Current;
        var fine = T(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4);
        var coarse = T(new[] { 1, 1, 1, 1 }, 10);
        Func<Tensor<double>, Tensor<double>> lateral = x => engine.TensorMultiplyScalar(x, 2.0);
        Func<Tensor<double>, Tensor<double>> output = x => engine.TensorAddScalar(x, 1.0);

        var levels = FeaturePyramidOps.TopDown(new[] { fine, coarse }, new[] { lateral, lateral }, new[] { output, output });

        // P_coarse = 2 * 10 = 20 (output 21); P_fine = 2 * [1, 2, 3, 4] + nearest(20) = [22, 24, 26, 28] (output + 1).
        Assert.Equal(2, levels.Count);
        AssertValues(levels[0], 23, 25, 27, 29);
        AssertValues(levels[1], 21);
    }

    [Fact]
    public void HungarianMatch_FindsTheMinimumTotalCostWhereGreedyWouldNot()
    {
        // Greedy row by row takes (0, 0) then (1, 1) for 11; the optimum is (0, 1) + (1, 0) = 3.
        var (segment, prediction) = MaskSetPredictionLoss.Match(T(new[] { 2, 3 }, 1, 2, 9, 1, 10, 9));

        Assert.Equal(new[] { 0, 1 }, segment);
        Assert.Equal(new[] { 1, 0 }, prediction);
    }

    [Fact]
    public void Mask2FormerLoss_MatchesTheBetterQueryAndWeighsClassBceAndDice()
    {
        // One class plus no-object, two queries, two pixels; the segment is class 0 on pixel 0. Class logits are
        // all zero; query 0's mask logits [2, -2] fit the segment, query 1's [-2, 2] do not, so query 0 is matched.
        var classLogits = T(new[] { 1, 2, 2 }, 0, 0, 0, 0);
        var maskLogits = T(new[] { 1, 2, 2 }, 2, -2, -2, 2);
        var segments = new[] { new MaskSetPredictionLoss.Segments<double>(T(new[] { 1, 2 }, 1, 0), new[] { 0 }) };

        var loss = MaskSetPredictionLoss.Mask2Former(classLogits, maskLogits, segments);

        // Class: every NLL is ln 2, so the 0.1-weighted mean is ln 2; times 2.
        // BCE of query 0: (softplus(2) - 2 + softplus(-2)) / 2 = 0.126928; times 5.
        // Dice of query 0: 1 - (2 sigmoid(2) + 1) / (sigmoid(2) + sigmoid(-2) + 1 + 1) = 0.0794689; times 5.
        AssertValues(loss, 2.4182775);
    }

    [Fact]
    public void KMaXLoss_CombinesPanopticQualityAndMaskIdCrossEntropy()
    {
        // Two clusters, one class plus no-object, two pixels; the segment is class 0 on pixel 0. Class logits are
        // zero (p = 0.5); cluster logits [ln 3, 0] and [0, 0], so the softmax masks are [0.75, 0.5] and [0.25, 0.5].
        var classLogits = T(new[] { 1, 2, 2 }, 0, 0, 0, 0);
        var maskLogits = T(new[] { 1, 2, 2 }, Math.Log(3), 0, 0, 0);
        var segments = new[] { new MaskSetPredictionLoss.Segments<double>(T(new[] { 1, 2 }, 1, 0), new[] { 0 }) };

        var loss = MaskSetPredictionLoss.KMaX(classLogits, maskLogits, segments);

        // Dice: cluster 0 = 1.5 / 2.25 = 2/3, cluster 1 = 0.5 / 1.75; similarity p * Dice matches cluster 0.
        // L_pos = -(0.5 * 2/3 + 2/3 * ln 0.5) = 0.1287651; L_neg = ln 2 (cluster 1 is no-object).
        // L_PQ = 0.75 L_pos + 0.25 L_neg; mask-ID CE = -ln 0.75 on the one segment pixel.
        // Total = 3 L_PQ + 0.3 CE.
        AssertValues(loss, 0.8958858);
    }

    [Fact]
    public void CascadeMaskHead_PastesEachRoiMaskIntoItsPixelBox()
    {
        var roiMasks = T(new[] { 2, 4 }, 1, 1, 1, 1, 1, 1, 1, 1);
        var boxes = T(new[] { 2, 4 }, 1, 1, 3, 3, 0.5, 0.5, 1.5, 2.2);

        var pasted = CascadeMaskRCNNHead<double>.PasteMasks(roiMasks, boxes, 4, 4).ToArray();

        // Box 0 covers rows and columns 1-2; box 1 is widened to whole pixels: columns 0-1, rows 0-2.
        Assert.Equal(new[]
        {
            0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        }, pasted);
    }

    [Fact]
    public void DescendingOrder_BreaksTiesByTheLowerIndex()
    {
        Assert.Equal(new[] { 1, 0, 2, 3 }, MaskClassificationOps.DescendingOrder(T(new[] { 4 }, 0.5, 0.9, 0.5, 0.1)));
    }

    private static readonly KMaXPanopticSettings KMaXDefaults = new(0.7, 0.5, 0.4, 0.8, 1.0, 1.0);

    [Fact]
    public void MergeKMaX_KeepsConfidentClustersAndMergesStuffOfOneClass()
    {
        // Class 0 is stuff, class 1 a thing; the last column is void.
        var classes = T(new[] { 3, 3 }, 0.9, 0.05, 0.05, 0.1, 0.8, 0.1, 0.6, 0.2, 0.2);
        // Softmax over the clusters, pixels p0..p3.
        var masks = T(new[] { 3, 1, 4 }, 0.8, 0.7, 0.1, 0.1, 0.1, 0.2, 0.8, 0.2, 0.1, 0.1, 0.1, 0.7);

        var result = MaskClassificationOps.MergeKMaX(classes, masks, new[] { false, true }, KMaXDefaults);

        // Order by score * mask score: q0 0.9*0.75, q1 0.8*0.8, q2 0.6*0.7. q0 takes p0-p1 (stuff, segment 1),
        // q1 takes p2 (thing, segment 2), q2 is stuff class 0 again and merges into segment 1.
        AssertValues(result.SemanticMap, 0, 0, 1, 0);
        AssertValues(result.InstanceMap, 0, 0, 2, 0);
        AssertValues(result.PanopticMap, 0, 0, 1002, 0);
        Assert.Equal(2, result.Segments.Count);
        Assert.False(result.Segments[0].IsThing);
        Assert.Equal(3, result.Segments[0].Area);
        Assert.Equal(0.9, result.Segments[0].Confidence, 6);
        Assert.True(result.Segments[1].IsThing);
        Assert.Equal(1, result.Segments[1].ClassId);
        Assert.Equal(0.8, result.Segments[1].Confidence, 6);
    }

    [Fact]
    public void MergeKMaX_DropsAThingBelowTheThingThreshold()
    {
        // q1's thing score 0.65 is under 0.7, so its pixel stays void (-1).
        var classes = T(new[] { 3, 3 }, 0.9, 0.05, 0.05, 0.2, 0.65, 0.15, 0.6, 0.2, 0.2);
        var masks = T(new[] { 3, 1, 4 }, 0.8, 0.7, 0.1, 0.1, 0.1, 0.2, 0.8, 0.2, 0.1, 0.1, 0.1, 0.7);

        var result = MaskClassificationOps.MergeKMaX(classes, masks, new[] { false, true }, KMaXDefaults);

        AssertValues(result.SemanticMap, 0, 0, -1, 0);
        AssertValues(result.PanopticMap, 0, 0, -1, 0);
        Assert.Single(result.Segments);
    }

    [Fact]
    public void MergeMask2Former_NeedsTheQueryToWinMostOfItsOwnMask()
    {
        var classes = T(new[] { 2, 3 }, 0.9, 0.05, 0.05, 0.05, 0.85, 0.1);
        var masks = T(new[] { 2, 1, 3 }, 0.9, 0.6, 0.2, 0.1, 0.7, 0.9);

        var result = MaskClassificationOps.MergeMask2Former(classes, masks, new[] { false, true }, 0.8, 0.8);

        // score * mask: q0 [0.81, 0.54, 0.18], q1 [0.085, 0.595, 0.765]; q0 wins only p0, half of its own mask
        // (p0, p1), so it is dropped; q1 wins p1 and p2, all of its own mask.
        AssertValues(result.SemanticMap, -1, 1, 1);
        AssertValues(result.InstanceMap, 0, 1, 1);
        AssertValues(result.PanopticMap, -1, 1001, 1001);
        Assert.Single(result.Segments);
        Assert.Equal(2, result.Segments[0].Area);
        Assert.Equal(0.85, result.Segments[0].Confidence, 6);
    }

    [Fact]
    public void MergePanopticFpn_PastesInstancesByScoreThenKeepsLargeStuff()
    {
        var semantic = new Tensor<int>(new[] { 0, 1, 2, 2 }, new[] { 4 });
        var probabilities = T(new[] { 3, 4 }, 0.7, 0.3, 0.1, 0.2, 0.2, 0.6, 0.1, 0.2, 0.1, 0.1, 0.8, 0.6);
        var instances = new Tensor<int>(new[] { 0, 0, 1, 1, 0, 0, 0, 1 }, new[] { 2, 4 });
        var scores = T(new[] { 2 }, 0.9, 0.3);
        var settings = new PanopticFpnSettings(0.4, 0.5, 2.0);

        var result = MaskClassificationOps.MergePanopticFpn(
            semantic, probabilities, instances, new[] { 2, 2 }, scores, new[] { false, false, true }, 1, 4, settings, pixelArea: 2.0);

        // The 0.3 instance is under the 0.4 threshold; each stuff region covers 1 map pixel = 2 input pixels.
        AssertValues(result.SemanticMap, 0, 1, 2, 2);
        AssertValues(result.InstanceMap, 0, 0, 1, 1);
        AssertValues(result.PanopticMap, 0, 1000, 2001, 2001);
        Assert.Equal(3, result.Segments.Count);
        Assert.Equal(0.9, result.Segments[0].Confidence, 6);
        Assert.Equal(0.7, result.Segments[1].Confidence, 6);
        Assert.Equal(0.6, result.Segments[2].Confidence, 6);
    }

    private static NeuralNetworkArchitecture<double> Architecture(int channels, int size, int classes) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: size, inputWidth: size, inputDepth: channels, outputSize: classes);

    [Fact]
    public void Odise_FusesTheDiffusionClassifierWithTheClipImageClassificationOfTheMaskedQuery()
    {
        var clip = new DeterministicVisualEncoder<double>(embeddingDimension: 2, channels: 2, imageSize: 2);
        var model = new ODISE<double>(Architecture(2, 32, 2), new DeterministicTextEncoder<double>(2), clip, options: new ODISEOptions
        {
            NumClasses = 2, NumQueries = 1, PixelEmbeddingDimension = 1, NumDecoderLayers = 1, Temperature = 0.5,
            InVocabularyClassNames = new[] { "class 0" },
        });
        // Two pixels a, b: P = [1, -2], query logits [1, -1], text-space embeddings (1, 0) and (0, 1), void 0.
        // The query attends only to a, so its embedding is (1, 0) and its logits become [2, -3]: the mask is {a}.
        var raw = T(new[] { 1, 5, 1, 2 }, 1, -2, 1, -1, 1, 0, 0, 1, 0, 0);
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 2, 2 }, 1, 0, 0, 1));
        // Pixel a is pure channel 0 and b pure channel 1, so the masked image embeds to (1, 0) in CLIP space.
        var image = T(new[] { 1, 2, 1, 2 }, 1, 0, 0, 1);
        var weights = model.ClipWeights(new[] { "class 0", "class 1" });

        var (_, trained) = model.ClassifyQueries(raw, text, weights, null);
        var (_, fused) = model.ClassifyQueries(raw, text, weights, image);

        // Diffusion logits: learned scale 1 / 0.5 = 2 times cos = [2, 0]; with the void logit 0 the training
        // distribution is softmax([2, 0, 0]) = [0.786986, 0.106507, 0.106507].
        AssertValues(trained, 0.7869860, 0.1065070, 0.1065070);
        // CLIP: softmax(100 * [1, 0]) = [1, e^-100]. In-vocabulary class 0: 0.880797^0.65 * 1^0.35; novel class 1:
        // 0.119203^0.35 * (e^-100)^0.65 ~ 1e-29, so after renormalization class 0 takes all of 1 - p_void.
        AssertValues(fused, 0.8934930, 0, 0.1065070);
        Assert.Equal(1, clip.EncodeCount);
    }

    [Fact]
    public void Odise_RejectsAClipImageEncoderOutsideTheTextSpace()
    {
        Assert.Throws<ArgumentException>(() => new ODISE<double>(Architecture(3, 32, 2), new DeterministicTextEncoder<double>(4),
            new DeterministicVisualEncoder<double>(embeddingDimension: 8), options: new ODISEOptions { NumClasses = 2 }));
    }

    [Fact]
    public void Cups_FusesPseudoLabelsAndSplitsThingsByInstanceCoverage()
    {
        var model = new CUPS<double>(Architecture(3, 32, 3), options: new CUPSOptions
        {
            NumClasses = 3, ChannelDimensions = new[] { 8, 8, 8, 8 }, StageDepths = new[] { 1, 1, 1, 1 }, DecoderDimension = 8,
        });
        var semantic = T(new[] { 2, 3 }, 0, 0, 1, 1, 2, 2);
        var instances = T(new[] { 1, 2, 3 }, 0, 0, 1, 0, 1, 1);

        var things = model.FitThingStuffSplit(new[] { semantic }, new[] { instances });

        // Coverage: class 0 0/2, class 1 1/2, class 2 2/2; psi_ts = 0.08.
        Assert.Equal(new[] { false, true, true }, things.ToArray());

        var (labels, ids) = model.GeneratePanopticPseudoLabels(semantic, instances);
        // The mask covers one class-1 and two class-2 pixels, so it is class 2; the uncovered class-1 pixel
        // is a thing pixel outside every instance and is ignored.
        AssertValues(labels, 0, 0, 2, -1, 2, 2);
        AssertValues(ids, 0, 0, 1, 0, 1, 1);
    }

    [Fact]
    public void Cups_InstanceTargetsTakeTheMajorityClassAndTheEnclosingBox()
    {
        var model = new CUPS<double>(Architecture(3, 32, 3), options: new CUPSOptions
        {
            NumClasses = 3, ChannelDimensions = new[] { 8, 8, 8, 8 }, StageDepths = new[] { 1, 1, 1, 1 }, DecoderDimension = 8,
        });
        // Instance 1 covers labels 1, 1, 2 (majority 1); instance 2 covers label 2; the -1 pixel is ignored.
        var semantic = T(new[] { 2, 3 }, 0, 1, 1, -1, 2, 2);
        var ids = T(new[] { 2, 3 }, 0, 1, 1, 0, 1, 2);

        var (boxes, classes, masks) = model.InstanceTargets(semantic, ids);

        Assert.Equal(new[] { 1, 2 }, classes.ToArray());
        Assert.Equal(new[] { 1.0, 0, 3, 2 }, boxes[0]);
        Assert.Equal(new[] { 2.0, 1, 3, 2 }, boxes[1]);
        Assert.Equal(new[] { 2, 1, 2, 3 }, masks.Shape.ToArray());
        AssertValues(masks, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
    }

    [Fact]
    public void Deva_ReadsMemoryWithTheTopKNegativeL2Affinity()
    {
        // Keys 0, 1, 3 against query 1: (2kq - k^2) = 0, 1, -3; top-2 keeps 0 and 1.
        var read = DEVA<double>.ReadMemory(T(new[] { 1, 3 }, 0, 1, 3), T(new[] { 1, 3 }, 1, 0, 0.5), T(new[] { 1, 1 }, 1), 2);

        AssertValues(read, 1.0 / (1.0 + Math.E));
    }

    [Fact]
    public void Deva_SoftAggregationAddsTheBackgroundAsTheProductOfComplements()
    {
        var logits = DEVA<double>.AggregateLogits(T(new[] { 1, 1 }, 1.0 / (1.0 + Math.E)), 1, 1);

        // Background 1 - p = e / (1 + e): logit 1; the object: logit -1.
        AssertValues(logits, 1, -1);
    }

    [Fact]
    public void Deva_AssociatesSegmentsByIou()
    {
        var iou = DEVA<double>.Iou(
            new Tensor<int>(new[] { 1, 1, 0 }, new[] { 1, 3 }), new Tensor<int>(new[] { 0, 1, 1, 1, 1, 0 }, new[] { 2, 3 }));

        AssertValues(iou, 1.0 / 3.0, 1.0);
    }

    [Fact]
    public void Deva_ForwardIsTheNetworkOutputWithoutAnInputResidual()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 2, inputWidth: 2, inputDepth: 2, outputSize: 2,
            layers: new List<ILayer<double>> { new ActivationLayer<double>((IActivationFunction<double>)new IdentityActivation<double>()) });
        var model = new DEVA<double>(architecture, options: new DEVAOptions { NumClasses = 2 });
        var input = T(new[] { 1, 2, 2, 2 }, 1, 2, 3, 4, 5, 6, 7, 8);

        // The removed residual added 1e-3 times the input's channel mean (e.g. 0.003 at the first pixel).
        AssertValues(model.Predict(input), 1, 2, 3, 4, 5, 6, 7, 8);
    }

    [Fact]
    public void UniVS_PromptQueryIsTheTokenMeanPlusPromptCrossAttention()
    {
        var query = UniVS<double>.PromptQuery(T(new[] { 2, 2 }, 2, 0, 0, 0));

        // q0 = (1, 0); attention softmax([2, 0] / sqrt(2)) = [0.804430, 0.195570]; q = q0 + 0.804430 * (2, 0).
        AssertValues(query, 2.6088594, 0);
    }

    [Fact]
    public void UniVS_VisualSamplerSpreadsPointsOverTheMask()
    {
        var points = UniVS<double>.SamplePromptPoints(T(new[] { 7 }, 0, 1, 1, 0, 1, 1, 1), 2);

        // Foreground positions 1, 2, 4, 5, 6; two points at ranks 0 and 2.
        Assert.NotNull(points);
        Assert.Equal(new[] { 1, 4 }, points?.ToArray());
        Assert.Null(UniVS<double>.SamplePromptPoints(T(new[] { 3 }, 0, 0.2, 0), 2));
    }

    [Fact]
    public void UniVS_PromptMemoryKeepsTheMostRecentPromptsOnly()
    {
        // Official UniVS: prompt_memory[:, -NUM_PREV_FRAMES_MEMORY:]; no prompt is pinned.
        var pool = new List<Tensor<double>>();
        var first = T(new[] { 1, 1 }, 1);
        var second = T(new[] { 1, 1 }, 2);
        var third = T(new[] { 1, 1 }, 3);

        UniVS<double>.AppendToMemory(pool, first, 2);
        UniVS<double>.AppendToMemory(pool, second, 2);
        UniVS<double>.AppendToMemory(pool, third, 2);

        Assert.Equal(2, pool.Count);
        Assert.Same(second, pool[0]);
        Assert.Same(third, pool[1]);
        Assert.Equal(5, new UniVSOptions().PromptMemoryLength);
    }

    private static UniverSeg<double> SmallUniverSeg() => new(Architecture(1, 8, 1), options: new UniverSegOptions
    {
        NumClasses = 1, EncoderFeatures = new[] { 4, 4 }, DecoderFeatures = new[] { 4 },
    });

    private static Tensor<double> Pattern(int seed, params int[] shape)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int length = shape.Aggregate(1, (a, b) => a * b);
        return T(shape, Enumerable.Range(0, length).Select(_ => random.NextDouble()).ToArray());
    }

    private static Tensor<double> Stack(params Tensor<double>[] parts)
    {
        var values = parts.SelectMany(p => p.ToArray()).ToArray();
        return T(new[] { parts.Length }.Concat(parts[0].Shape.ToArray()).ToArray(), values);
    }

    [Fact]
    public void UniverSeg_AveragesTheCrossBlockInteractionOverTheSupportSet()
    {
        var model = SmallUniverSeg();
        var query = Pattern(1, 1, 8, 8);
        var imageA = Pattern(2, 1, 8, 8);
        var imageB = Pattern(3, 1, 8, 8);
        var labelA = Pattern(4, 8, 8);
        var labelB = Pattern(5, 8, 8);

        model.SetSupportSet(Stack(imageA, imageB), Stack(labelA, labelB));
        var ab = model.Predict(query).ToArray();
        model.SetSupportSet(Stack(imageB, imageA), Stack(labelB, labelA));
        var ba = model.Predict(query).ToArray();
        model.SetSupportSet(Stack(imageA), Stack(labelA));
        var a = model.Predict(query).ToArray();
        model.SetSupportSet(Stack(imageA, imageA), Stack(labelA, labelA));
        var aa = model.Predict(query).ToArray();

        // A mean over the support set is invariant to its order and to duplicating every entry; a sum or a
        // single-support path would not be.
        Assert.Equal(64, ab.Length);
        for (int i = 0; i < ab.Length; i++)
        {
            Assert.Equal(ab[i], ba[i], 9);
            Assert.Equal(a[i], aa[i], 9);
        }

        Assert.Contains(Enumerable.Range(0, ab.Length), i => Math.Abs(ab[i] - a[i]) > 1e-9);
    }

    [Fact]
    public void UniverSeg_RejectsPredictionWithoutASupportSet()
    {
        // The paper defines f(x, S) only for |S| >= 1; there is no zero-support prediction.
        var model = SmallUniverSeg();
        var query = Pattern(1, 1, 8, 8);

        Assert.Throws<InvalidOperationException>(() => model.Predict(query));
        model.SetSupportSet(Stack(Pattern(2, 1, 8, 8)), Stack(Pattern(4, 8, 8)));
        Assert.Equal(64, model.Predict(query).Length);
        model.ClearSupportSet();
        Assert.Throws<InvalidOperationException>(() => model.Predict(query));
    }

    // ---- Review fixes: Half counts, overflow and clamps, detectron2 candidates, UniverSeg upsampling ----

    private static Tensor<Half> H(int[] shape, IEnumerable<double> values)
    {
        var half = MathHelper.GetNumericOperations<Half>();
        return new Tensor<Half>(shape, new Vector<Half>(values.Select(v => half.FromDouble(v)).ToArray()));
    }

    private static double[] D(Tensor<Half> tensor)
    {
        var half = MathHelper.GetNumericOperations<Half>();
        return tensor.ToArray().Select(v => half.ToDouble(v)).ToArray();
    }

    [Fact]
    public void DescendingOrder_HalfRanksPast2048ByIntegerIndex()
    {
        // 2100 equal scores: the order is the index order. In Half, indices 2049 and 2050 both round to 2050 and a
        // rank summed in Half stops at 2048, so ranks collided and the order was rejected.
        const int n = 2100;
        int[] order = MaskClassificationOps.DescendingOrder(H(new[] { n }, Enumerable.Repeat(0.5, n)));

        Assert.Equal(Enumerable.Range(0, n).ToArray(), order);
    }

    [Fact]
    public void NormalizeAxis_HalfComponentsAbove256DoNotOverflow()
    {
        // (300, 400) / 500 = (0.6, 0.8). Squaring first gives 90000 and 160000, past Half's 65504, and a zero vector.
        var normalized = D(MaskClassificationOps.NormalizeAxis(H(new[] { 2, 1 }, new double[] { 300, 400 }), 0));

        Assert.Equal(0.6, normalized[0], 3);
        Assert.Equal(0.8, normalized[1], 3);
    }

    [Fact]
    public void MergeKMaX_HalfCountsEachMaskInInt()
    {
        // 5000 pixels, one thing class. Cluster A (score 0.95, mask 0.9 on pixels 0-2999) is visited first and
        // claims its 3000 pixels. Cluster B (score 0.9, mask 0.8 on all 5000) then has 2000 free pixels, not more
        // than 0.8 * 5000 = 4000, so it is dropped. Counted in Half its mask held 2048 pixels and 2000 > 1638.4
        // let it through as a second segment.
        const int hw = 5000;
        var classes = H(new[] { 2, 2 }, new[] { 0.95, 0.05, 0.9, 0.1 });
        var masks = H(new[] { 2, 1, hw },
            Enumerable.Range(0, hw).Select(p => p < 3000 ? 0.9 : 0.0).Concat(Enumerable.Repeat(0.8, hw)));

        var result = MaskClassificationOps.MergeKMaX(classes, masks, new[] { true }, KMaXDefaults);

        Assert.Single(result.Segments);
        Assert.Equal(3000, result.Segments[0].Area);
    }

    [Fact]
    public void VideoStatistics_HalfMeanOverMoreThan2048Pixels()
    {
        // 3000 pixels at 0.75: the confidence is 0.75. Summed in Half the area stopped at 2048 and the
        // probability sum near 2048, so the mean came out near 1.
        var (confidences, visible) = VideoTrackingOps.Statistics(H(new[] { 1, 50, 60 }, Enumerable.Repeat(0.75, 3000)));

        Assert.Equal(0.75, confidences[0], 3);
        Assert.True(visible[0]);
    }

    [Fact]
    public void Deva_IouCountsPast2048InInt()
    {
        // A covers pixels 0-2999, B pixels 0-3999: IoU = 3000 / 4000. In Half both areas stopped at 2048.
        const int hw = 5000;
        var a = new Tensor<int>(Enumerable.Range(0, hw).Select(p => p < 3000 ? 1 : 0).ToArray(), new[] { 1, hw });
        var b = new Tensor<int>(Enumerable.Range(0, hw).Select(p => p < 4000 ? 1 : 0).ToArray(), new[] { 1, hw });

        Assert.Equal(0.75, D(DEVA<Half>.Iou(a, b))[0], 3);
    }

    [Fact]
    public void Deva_SoftAggregationStaysFiniteInHalf()
    {
        // A certain object (p = 1): XMem clamps to 1 - 1e-7, which Half rounds to 1, so the logit log(p / (1 - p))
        // was +inf and the background's -inf. The clamp now stays strictly inside (0, 1).
        var logits = D(DEVA<Half>.AggregateLogits(H(new[] { 1, 1 }, new[] { 1.0 }), 1, 1));

        Assert.All(logits, value => Assert.False(double.IsInfinity(value) || double.IsNaN(value)));
        Assert.True(logits[1] > 7.0);
        Assert.True(logits[0] < -7.0);
    }

    [Fact]
    public void UniVS_PromptQueryScalesBeforeTheProductInHalf()
    {
        // Two identical 64-wide tokens of 40: q0 = 40, the attention is uniform, q = 40 + 40 = 80. The unscaled
        // product 64 * 40 * 40 = 102400 overflows Half to inf and the softmax gave NaN.
        var query = D(UniVS<Half>.PromptQuery(H(new[] { 2, 64 }, Enumerable.Repeat(40.0, 128))));

        Assert.All(query, value => Assert.Equal(80.0, value, 1));
    }

    [Fact]
    public void Cups_MajorityClassCountsPast2048InInt()
    {
        var model = new CUPS<Half>(new NeuralNetworkArchitecture<Half>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options: new CUPSOptions
        {
            NumClasses = 3, ChannelDimensions = new[] { 8, 8, 8, 8 }, StageDepths = new[] { 1, 1, 1, 1 }, DecoderDimension = 8,
        });
        // One mask over 5500 pixels: 2500 of class 1, then 3000 of class 2, so its class is 2. As a Half matrix
        // product both counts stopped at 2048, tied, and the argmax took class 1.
        const int hw = 5500;
        var semantic = H(new[] { 50, 110 }, Enumerable.Range(0, hw).Select(p => p < 2500 ? 1.0 : 2.0));
        var masks = H(new[] { 1, 50, 110 }, Enumerable.Repeat(1.0, hw));

        var (labels, ids) = model.GeneratePanopticPseudoLabels(semantic, masks);

        Assert.All(D(labels), value => Assert.Equal(2.0, value));
        Assert.All(D(ids), value => Assert.Equal(1.0, value));
    }

    [Fact]
    public void CascadeHead_KeepsEveryClassAboveTheScoreThresholdOfARoi()
    {
        // detectron2's fast_rcnn_inference keeps each (RoI, class) pair scoring above 0.05, so one RoI scoring 0.6
        // for class 0 and 0.3 for class 1 gives two candidates; the argmax-only version kept class 0 alone. A score
        // equal to the threshold is not above it.
        var foreground = T(new[] { 1, 3 }, 0.6, 0.3, 0.05);
        var boxes = T(new[] { 1, 4 }, 1, 2, 5, 6);

        var candidates = CascadeMaskRCNNHead<double>.Candidates(foreground, boxes, 0.05);

        Assert.Equal(new[] { 0, 1 }, candidates.Select(c => c.ClassId).ToArray());
        Assert.Equal(new[] { 0.6, 0.3 }, candidates.Select(c => c.Confidence).ToArray());
        Assert.All(candidates, c => Assert.Equal(new[] { 1.0, 2, 5, 6 }, new[] { c.Box.X1, c.Box.Y1, c.Box.X2, c.Box.Y2 }));
    }

    [Fact]
    public void UniverSeg_DecoderUpsamplesWithAlignedCorners()
    {
        // nn.UpsamplingBilinear2d (align_corners=True): [0, 1] -> [0, 1/3, 2/3, 1]. Half-pixel centres give
        // [0, 0.25, 0.75, 1].
        var upsampled = UniverSeg<double>.Upsample(T(new[] { 1, 1, 1, 2 }, 0, 1), new[] { 1, 4 });

        AssertValues(upsampled, 0, 1.0 / 3.0, 2.0 / 3.0, 1);
    }
}
