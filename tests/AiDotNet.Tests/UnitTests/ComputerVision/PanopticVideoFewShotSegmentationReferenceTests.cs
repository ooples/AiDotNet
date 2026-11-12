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
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Hand-derived references for the paper rewrites of kMaX-DeepLab (k-means cross-attention and panoptic merge),
/// ODISE (Mask2Former merge and the in/out-of-vocabulary fusion), CUPS (pseudo-label fusion, thing/stuff split,
/// Panoptic FPN merge), DEVA (memory reading, soft aggregation, no input residual), UniVS (prompt memory) and
/// UniverSeg (CrossBlock averaged over the support set).
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
    public void KMeansCrossAttention_AssignsEachPixelToItsArgmaxClusterAndSumsItsEmbedding()
    {
        // D = 2, N = 2, K + 1 = 2 classes, pixels p0 = (3, 4) -> (0.6, 0.8) and p1 = (1, 0).
        // Initial cluster logits n0 = [1, 0], n1 = [0, 1]; class projections z(p0) = (2, 0), z(p1) = (0, 3).
        var raw = T(new[] { 1, 6, 1, 2 },
            3, 1, 4, 0,   // pixel embedding channels
            1, 0, 0, 1,   // cluster logits
            2, 0, 0, 3);  // class projections

        var (logits, classes) = MaskClassificationOps.KMeansCrossAttention(raw, pixelDim: 2, numClusters: 2, iterations: 2);

        // Each update: p0 -> n0, p1 -> n1, so C_n0 += p0, C_n1 += p1 and L += C . P:
        // n0 gains [p0.p0, p0.p1] = [1, 0.6]; n1 gains [p1.p0, p1.p1] = [0.6, 1]; twice.
        Assert.Equal(new[] { 1, 2, 2 }, logits.Shape.ToArray());
        AssertValues(logits, 3, 1.2, 1.2, 3);
        // Class logits sum the class projections of the assigned pixels over both updates.
        AssertValues(classes, 4, 0, 0, 6);
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
    public void Odise_FusesDiffusionAndPooledPredictionsWithTheVocabularyWeights()
    {
        var model = new ODISE<double>(Architecture(3, 32, 2), new DeterministicTextEncoder<double>(2), options: new ODISEOptions
        {
            NumClasses = 2, NumQueries = 1, PixelEmbeddingDimension = 1, NumDecoderLayers = 1, Temperature = 0.5,
            InVocabularyClassNames = new[] { "class 0" },
        });
        // Two pixels a, b: P = [1, 2], query logits [1, -1], text-space embeddings (1, 0) and (0, 1), void 0.
        // The query attends only to a (b is outside its mask), so its embedding is (1, 0); after the update
        // its logits are [2, 1], so the mask-pooled embedding is (0.5, 0.5).
        var raw = T(new[] { 1, 5, 1, 2 }, 1, 2, 1, -1, 1, 0, 0, 1, 0, 0);
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 2, 2 }, 1, 0, 0, 1));

        var (_, probabilities) = model.ClassifyQueries(raw, text, model.ClipWeights(new[] { "class 0", "class 1" }));

        // p_diffusion = softmax(2 * [1, 0]) = [0.880797, 0.119203]; p_clip = [0.5, 0.5]; p_void = 0.106507.
        // In-vocabulary class 0: 0.880797^0.65 * 0.5^0.35 = 0.722452; novel class 1: 0.119203^0.35 * 0.5^0.65
        // = 0.302713. Renormalized and scaled by 1 - p_void.
        AssertValues(probabilities, 0.6296604, 0.2638326, 0.1065070);
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
        var iou = DEVA<double>.Iou(T(new[] { 1, 3 }, 1, 1, 0), T(new[] { 2, 3 }, 0, 1, 1, 1, 1, 0));

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
}
