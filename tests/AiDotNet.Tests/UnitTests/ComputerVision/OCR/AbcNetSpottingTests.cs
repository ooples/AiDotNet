using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision.OCR;

/// <summary>
/// Verifies ABCNet's composition and its CTC decode (Liu et al., CVPR 2020, arXiv:2002.10200).
/// </summary>
/// <remarks>
/// The two contributions themselves are covered by <see cref="BezierTextRepresentationTests"/>. These
/// cover what this class adds on top: the single-shot detection layout, offsets decoded relative to their
/// own feature position, and the CTC collapse order — all checkable on an untrained model.
/// </remarks>
public class AbcNetSpottingTests
{
    private static ABCNetOptions<double> Options(int size = 32) => new()
    {
        InputHeight = size,
        InputWidth = size,
        InputChannels = 3,
        FeatureChannels = 16,
        FeatureStride = 4,
        BezierSampleHeight = 4,
        BezierSampleWidth = 8,
        NumCharacterClasses = 5,
        ConfidenceThreshold = 0.0,
        MaxInstances = 6,
    };

    private static Tensor<double> Image(int size, int seed)
    {
        var t = new Tensor<double>(new[] { 3, size, size });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
        return t;
    }

    [Fact]
    public void DetectionStacksTheScoreMapWithSixteenCoordinateChannels()
    {
        // The single-shot layout: one score channel plus 8 control points x 2 coordinates, over a
        // feature map at 1/FeatureStride resolution. 16 rather than 8 is the easy slip — the paper's
        // "8 control points" are POINTS, not numbers.
        var options = Options(32);
        var model = new ABCNet<double>(options);

        var detection = model.Predict(Image(32, 1));

        Assert.Equal(16, ABCNet<double>.BezierCoordinateCount);
        Assert.Equal(new[] { 1 + 16, 32 / 4, 32 / 4 }, detection.Shape.ToArray());
        Assert.Equal(8, model.FeatureHeight);
        Assert.Equal(8, model.FeatureWidth);
    }

    [Fact]
    public void ABranchedForwardRefusesAWrongLengthCustomLayerList()
    {
        // The forward pass routes layers by POSITION because it branches, so a custom list of the wrong
        // length cannot be padded or truncated — doing so would silently compute something else while
        // still training and predicting.
        Assert.Equal(8, ABCNet<double>.ExpectedLayerCount);

        var options = Options(32);
        var tooFew = new List<ILayer<double>> { new ConvolutionalLayer<double>(4, 3, 1, 1) };
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 17,
            layers: tooFew);

        Assert.Throws<ArgumentException>(() => new ABCNet<double>(options, arch));
    }

    [Fact]
    public void DetectedInstancesAreScoreOrderedAndCapped()
    {
        // With the threshold at 0 every one of the 8x8 positions qualifies, so the cap and the ordering
        // are both exercised. An uncapped head would run the recognition branch 64 times here, which is
        // the cost the cap exists to bound.
        var options = Options(32);
        options.MaxInstances = 6;
        var model = new ABCNet<double>(options);

        var instances = model.DetectInstances(Image(32, 2));

        Assert.Equal(6, instances.Count);
        for (int i = 1; i < instances.Count; i++)
            Assert.True(instances[i - 1].Score >= instances[i].Score, "Instances are not score-ordered.");

        foreach (var instance in instances)
        {
            Assert.Equal(BezierAlign.ControlPointCount, instance.ControlPoints.Count);
            foreach (var (x, y) in instance.ControlPoints)
            {
                Assert.False(double.IsNaN(x) || double.IsInfinity(x), "A control point x is not finite.");
                Assert.False(double.IsNaN(y) || double.IsInfinity(y), "A control point y is not finite.");
            }
        }
    }

    [Fact]
    public void RaisingTheThresholdRemovesInstances()
    {
        // Guards against a threshold that is read but never applied — the head's sigmoid keeps scores
        // near 0.5 on an untrained model, so a threshold above that must empty the result.
        var options = Options(32);
        options.ConfidenceThreshold = 1.5;   // unreachable through a sigmoid
        var model = new ABCNet<double>(options);

        Assert.Empty(model.DetectInstances(Image(32, 3)));
    }

    [Fact]
    public void SpottingReturnsCharactersBoundedByTheRectifiedWidth()
    {
        // End-to-end wiring: BezierAlign feeds the recognition branch, which emits one classifier output
        // per rectified COLUMN, so the decoded length cannot exceed that width.
        var options = Options(32);
        options.MaxInstances = 2;
        var model = new ABCNet<double>(options);

        var spotted = model.Spot(Image(32, 4));

        Assert.Equal(2, spotted.Count);
        foreach (var instance in spotted)
        {
            Assert.NotNull(instance.CharacterIndices);
            Assert.True(instance.CharacterIndices.Count <= options.BezierSampleWidth,
                $"Decoded {instance.CharacterIndices.Count} characters from only "
                + $"{options.BezierSampleWidth} rectified columns.");
            Assert.All(instance.CharacterIndices, k =>
                Assert.InRange(k, 1, options.NumCharacterClasses - 1));   // never the blank
        }
    }

    [Fact]
    public void RecognitionEmitsOneLogitVectorPerRectifiedColumn()
    {
        var options = Options(32);
        var model = new ABCNet<double>(options);

        var features = new Tensor<double>(new[] { options.FeatureChannels, 8, 8 });
        for (int i = 0; i < features.Length; i++) features[i] = 0.01 * i;

        // A straight instance well inside the 8x8 feature map.
        var cp = new Tensor<double>(new[] { 8, 2 });
        for (int k = 0; k < 4; k++) { cp[(k * 2) + 0] = 1.0 + k; cp[(k * 2) + 1] = 2.0; }
        for (int k = 0; k < 4; k++) { cp[((4 + k) * 2) + 0] = 1.0 + k; cp[((4 + k) * 2) + 1] = 5.0; }

        var logits = model.RecognizeRectified(features, cp);

        Assert.Equal(new[] { options.BezierSampleWidth, options.NumCharacterClasses }, logits.Shape.ToArray());
    }

    // ---------------- CTC decode ----------------

    /// <summary>Builds logits whose per-column argmax is the given class sequence.</summary>
    private static Tensor<double> LogitsFor(IReadOnlyList<int> argmaxPerColumn, int classes)
    {
        var t = new Tensor<double>(new[] { argmaxPerColumn.Count, classes });
        for (int c = 0; c < argmaxPerColumn.Count; c++)
            for (int k = 0; k < classes; k++)
                t[(c * classes) + k] = k == argmaxPerColumn[c] ? 1.0 : 0.0;
        return t;
    }

    [Fact]
    public void CtcCollapsesRepeatsBeforeDroppingBlanksSoDoubleLettersSurvive()
    {
        // THE test for the collapse order, and the reason it is not interchangeable. A blank between two
        // identical labels is exactly how CTC encodes a double letter. Dropping blanks first turns
        // [a, blank, a] into [a, a] and then collapses it to [a], deleting a letter from every word with
        // a double in it — silently, and only for those words.
        const int classes = 3;   // 0 = blank, 1 = 'a', 2 = 'b'

        Assert.Equal(new[] { 1, 1 }, ABCNet<double>.CtcGreedyDecode(LogitsFor(new[] { 1, 0, 1 }, classes)).ToArray());
        Assert.Equal(new[] { 1 }, ABCNet<double>.CtcGreedyDecode(LogitsFor(new[] { 1, 1 }, classes)).ToArray());
        Assert.Equal(new[] { 1, 2, 2 }, ABCNet<double>.CtcGreedyDecode(LogitsFor(new[] { 1, 2, 2, 0, 2 }, classes)).ToArray());
    }

    [Fact]
    public void CtcDropsBlanksAndAnAllBlankColumnSetDecodesToNothing()
    {
        const int classes = 3;
        Assert.Empty(ABCNet<double>.CtcGreedyDecode(LogitsFor(new[] { 0, 0, 0 }, classes)));
        Assert.Equal(new[] { 2 }, ABCNet<double>.CtcGreedyDecode(LogitsFor(new[] { 0, 2, 0 }, classes)).ToArray());
    }

    [Fact]
    public void CtcRejectsLogitsThatAreNotPerColumn()
    {
        Assert.Throws<ArgumentException>(() =>
            ABCNet<double>.CtcGreedyDecode(new Tensor<double>(new[] { 2, 3, 4 })));
    }
}
