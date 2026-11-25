using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Pins the whole-tensor segmentation post-processing (<see cref="SegmentationPostProcessing"/>) to the
/// pixel-by-pixel scans it replaced in the panoptic, YOLO-seg and promptable models. Every expected value is
/// produced by the original scan, written out below with plain loops.
/// </summary>
public class SegmentationPostProcessingParityTests
{
    private static T[] Flat<T>(Tensor<T> tensor) => tensor.ToArray();

    private static double D<T>(T value) => MathHelper.GetNumericOperations<T>().ToDouble(value);

    /// <summary>
    /// Class logits <c>[c, h, w]</c> whose winning class is constant over 3 x 4 blocks (with a few single-pixel
    /// exceptions, so components of area 1 exist) and whose margins vary per pixel.
    /// </summary>
    private static Tensor<T> BlockLogits<T>(int c, int h, int w, int seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new T[c * h * w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int winner = (y / 3 + 2 * (x / 4)) % c;
                if (random.NextDouble() < 0.08) winner = random.Next(c);
                for (int ch = 0; ch < c; ch++)
                {
                    double value = (ch == winner ? 2.5 : 0.0) + random.NextDouble();
                    data[(ch * h + y) * w + x] = ops.FromDouble(value);
                }
            }
        }

        return new Tensor<T>(data, new[] { c, h, w });
    }

    private static Tensor<T> RandomTensor<T>(int[] shape, int seed, double low, double high)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.CreateSeededRandom(seed);
        int length = shape.Aggregate(1, (a, b) => a * b);
        var data = new T[length];
        for (int i = 0; i < length; i++) data[i] = ops.FromDouble(low + (high - low) * random.NextDouble());
        return new Tensor<T>(data, shape);
    }

    private static void AssertSame<T>(Tensor<T> expected, Tensor<T> actual, string what)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = Flat(expected);
        var a = Flat(actual);
        for (int i = 0; i < e.Length; i++)
            Assert.True(D(e[i]).Equals(D(a[i])), $"{what}: element {i} expected {D(e[i]):R} but was {D(a[i]):R}");
    }

    // ------------------------------------------------------------------ panoptic

    private sealed class ReferenceSegment
    {
        public int PanopticValue;
        public int ClassId;
        public bool IsThing;
        public double Confidence;
        public int Area;
        public double[]? Mask;
    }

    /// <summary>The scan every SegmentPanoptic override ran, with the segment keyed by its panoptic value.</summary>
    private static (double[] Instance, double[] Panoptic, List<ReferenceSegment> Segments) ReferencePanoptic<T>(
        Tensor<T> logits, int numStuff, int numClasses)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var probMap = SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = semanticMap.Shape[0], w = semanticMap.Shape[1];
        var instanceMap = new double[h * w];
        var panopticMap = new double[h * w];
        var segments = new List<ReferenceSegment>();
        int nextInstId = 1;
        for (int cls = 0; cls < numStuff; cls++)
        {
            int area = 0;
            double sumConf = 0;
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    if (ops.Compare(semanticMap[row, col], ops.FromDouble(cls)) == 0)
                    {
                        panopticMap[row * w + col] = D(ops.FromDouble(cls * 1000));
                        area++;
                        sumConf += ops.ToDouble(probMap[cls, row, col]);
                    }
                }
            }

            if (area > 0)
                segments.Add(new ReferenceSegment { PanopticValue = cls * 1000, ClassId = cls, IsThing = false, Confidence = sumConf / area, Area = area });
        }

        for (int cls = numStuff; cls < numClasses; cls++)
        {
            var (labelMap, count) = SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            for (int comp = 1; comp <= count; comp++)
            {
                int instId = nextInstId++;
                int area = 0;
                double sumConf = 0;
                var compMask = new double[h * w];
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        if (ops.Compare(labelMap[row, col], ops.FromDouble(comp)) == 0)
                        {
                            instanceMap[row * w + col] = D(ops.FromDouble(instId));
                            panopticMap[row * w + col] = D(ops.FromDouble(cls * 1000 + instId));
                            compMask[row * w + col] = 1.0;
                            area++;
                            sumConf += ops.ToDouble(probMap[cls, row, col]);
                        }
                    }
                }

                if (area > 0)
                    segments.Add(new ReferenceSegment { PanopticValue = cls * 1000 + instId, ClassId = cls, IsThing = true, Confidence = sumConf / area, Area = area, Mask = compMask });
            }
        }

        return (instanceMap, panopticMap, segments);
    }

    private static void PanopticMatchesScan<T>(int classes, int numStuff, int numClasses, int seed)
    {
        var logits = BlockLogits<T>(classes, 12, 16, seed);
        var (instance, panoptic, segments) = ReferencePanoptic(logits, numStuff, numClasses);

        var result = SegmentationPostProcessing.PanopticFromSemanticLogits(logits, numStuff, numClasses);

        AssertSame(SegmentationTensorOps.ArgmaxAlongClassDim(logits), result.SemanticMap, "semantic map");
        Assert.Equal(instance, Flat(result.InstanceMap).Select(D).ToArray());
        Assert.Equal(panoptic, Flat(result.PanopticMap).Select(D).ToArray());
        Assert.Equal(new[] { 12, 16 }, result.PanopticMap.Shape.ToArray());

        Assert.NotEmpty(segments);
        Assert.Equal(segments.Count, result.Segments.Count);
        for (int i = 0; i < segments.Count; i++)
        {
            var expected = segments[i];
            var actual = result.Segments[i];
            Assert.Equal(expected.ClassId, actual.ClassId);
            Assert.Equal(expected.IsThing, actual.IsThing);
            Assert.Equal(expected.Area, actual.Area);
            // Same pixels added in the same (row-major) order, so the sum is the same double
            Assert.Equal(expected.Confidence, actual.Confidence);
            Assert.Equal(expected.PanopticValue, actual.SegmentId);
            if (expected.Mask is { } mask)
                Assert.Equal(mask, Flat(actual.Mask).Select(D).ToArray());
        }
    }

    [Theory]
    [InlineData(6, 2, 6, 11)]
    [InlineData(6, 0, 6, 12)]
    [InlineData(6, 6, 6, 13)]
    [InlineData(5, 2, 4, 14)]
    [InlineData(4, 2, 7, 15)]
    public void PanopticFromSemanticLogits_MatchesThePixelScan_Double(int classes, int numStuff, int numClasses, int seed)
        => PanopticMatchesScan<double>(classes, numStuff, numClasses, seed);

    [Theory]
    [InlineData(6, 2, 6, 21)]
    [InlineData(5, 1, 5, 22)]
    public void PanopticFromSemanticLogits_MatchesThePixelScan_Float(int classes, int numStuff, int numClasses, int seed)
        => PanopticMatchesScan<float>(classes, numStuff, numClasses, seed);

    [Fact]
    public void PanopticFromSemanticLogits_SegmentIdsAreUniqueAndAreThePanopticMapValues()
    {
        // Nine of the ten models used the class id for stuff and the instance id for things as SegmentId, so
        // stuff class 1 and instance 1 shared an id; the id is now the segment's value in PanopticMap.
        var logits = BlockLogits<double>(6, 12, 16, 31);
        var result = SegmentationPostProcessing.PanopticFromSemanticLogits(logits, 2, 6);

        var ids = result.Segments.Select(s => s.SegmentId).ToArray();
        Assert.Equal(ids.Length, ids.Distinct().Count());
        Assert.Contains(result.Segments, s => !s.IsThing);
        Assert.Contains(result.Segments, s => s.IsThing);

        var panoptic = Flat(result.PanopticMap);
        var counts = panoptic.GroupBy(v => (int)v).ToDictionary(g => g.Key, g => g.Count());
        foreach (var segment in result.Segments)
        {
            Assert.True(counts.TryGetValue(segment.SegmentId, out int pixels), $"no pixel carries id {segment.SegmentId}");
            Assert.Equal(segment.Area, pixels);
            Assert.Equal(segment.ClassId, segment.SegmentId / SegmentationPostProcessing.PanopticLabelDivisor);
        }

        Assert.Equal(panoptic.Length, result.Segments.Sum(s => s.Area));
    }

    // ------------------------------------------------------------------ instances

    private sealed class ReferenceInstance
    {
        public int ClassId;
        public double Confidence;
        public double[] Box = Array.Empty<double>();
        public double[] Mask = Array.Empty<double>();
    }

    /// <summary>
    /// The scan every YOLO-seg DetectInstances override ran, and the stable descending-confidence ordering it
    /// applied before its mask NMS.
    /// </summary>
    private static List<ReferenceInstance> ReferenceInstances<T>(Tensor<T> logits, int numClasses, int maxInstances, double threshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var probMap = SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var classMap = SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var instances = new List<ReferenceInstance>();
        for (int cls = 1; cls < numClasses && instances.Count < maxInstances; cls++)
        {
            var (labelMap, count) = SegmentationTensorOps.LabelConnectedComponents(classMap, cls);
            for (int comp = 1; comp <= count && instances.Count < maxInstances; comp++)
            {
                var mask = new double[h * w];
                int area = 0;
                double sumConf = 0;
                int minX = w, minY = h, maxX = 0, maxY = 0;
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        if (ops.Compare(labelMap[row, col], ops.FromDouble(comp)) == 0)
                        {
                            mask[row * w + col] = 1.0;
                            area++;
                            sumConf += ops.ToDouble(probMap[cls, row, col]);
                            if (col < minX) minX = col;
                            if (col > maxX) maxX = col;
                            if (row < minY) minY = row;
                            if (row > maxY) maxY = row;
                        }
                    }
                }

                if (area < 4) continue;
                double confidence = sumConf / area;
                if (confidence < threshold) continue;
                instances.Add(new ReferenceInstance
                {
                    ClassId = cls,
                    Confidence = D(ops.FromDouble(confidence)),
                    Box = new double[] { minX, minY, maxX + 1, maxY + 1 },
                    Mask = mask,
                });
            }
        }

        return instances.OrderByDescending(i => i.Confidence).ToList();
    }

    private static void InstancesMatchScan<T>(int numClasses, int maxInstances, double threshold, int seed)
    {
        var logits = BlockLogits<T>(5, 12, 16, seed);
        var expected = ReferenceInstances(logits, numClasses, maxInstances, threshold);

        var actual = SegmentationPostProcessing.ConnectedComponentInstances(
            logits, numClasses, maxInstances, minArea: 4, confidenceThreshold: threshold);

        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i].ClassId, actual[i].ClassId);
            Assert.Equal(expected[i].ClassId, actual[i].Box.ClassIndex);
            Assert.Equal(expected[i].Confidence, D(actual[i].Confidence));
            Assert.Equal(expected[i].Box, new[] { D(actual[i].Box.X1), D(actual[i].Box.Y1), D(actual[i].Box.X2), D(actual[i].Box.Y2) });
            Assert.Equal(new[] { 12, 16 }, actual[i].Mask.Shape.ToArray());
            Assert.Equal(expected[i].Mask, Flat(actual[i].Mask).Select(D).ToArray());
        }
    }

    [Theory]
    [InlineData(5, 100, 0.5, 41)]
    [InlineData(5, 100, 0.0, 42)]
    [InlineData(5, 3, 0.0, 43)]
    [InlineData(3, 100, 0.5, 44)]
    [InlineData(9, 100, 0.5, 45)]
    [InlineData(5, 100, 0.99, 46)]
    public void ConnectedComponentInstances_MatchesThePixelScan_Double(int numClasses, int maxInstances, double threshold, int seed)
        => InstancesMatchScan<double>(numClasses, maxInstances, threshold, seed);

    [Theory]
    [InlineData(5, 100, 0.5, 51)]
    [InlineData(5, 2, 0.0, 52)]
    public void ConnectedComponentInstances_MatchesThePixelScan_Float(int numClasses, int maxInstances, double threshold, int seed)
        => InstancesMatchScan<float>(numClasses, maxInstances, threshold, seed);

    [Fact]
    public void ConnectedComponentInstances_KeepsSomethingAndDropsSomething()
    {
        // Guards the parity cases above against a fixture that yields nothing to compare
        var logits = BlockLogits<double>(5, 12, 16, 41);
        Assert.NotEmpty(ReferenceInstances(logits, 5, 100, 0.5));
        Assert.True(ReferenceInstances(logits, 5, 100, 0.0).Count > ReferenceInstances(logits, 5, 3, 0.0).Count);
        Assert.Empty(ReferenceInstances(logits, 5, 100, 0.99));
    }

    // ------------------------------------------------------------------ point prompts

    private static void PointAttentionMatchesScan<T>(double? positiveAtOrAbove)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int h = 7, w = 9;
        const double sigma = 0.9;
        var points = new Tensor<T>(
            new[] { 2.0, 3.0, 6.5, 1.25, 0.0, 6.0, 8.0, 0.0 }.Select(ops.FromDouble).ToArray(), new[] { 4, 2 });
        var labels = new Tensor<T>(new[] { 1.0, 0.0, 0.75, 2.0 }.Select(ops.FromDouble).ToArray(), new[] { 4 });

        var expected = new Tensor<T>(new[] { h, w });
        for (int i = 0; i < 4; i++)
        {
            double px = ops.ToDouble(points[i, 0]), py = ops.ToDouble(points[i, 1]);
            double sign = positiveAtOrAbove is { } floor
                ? (ops.GreaterThanOrEquals(labels[i], ops.FromDouble(floor)) ? 1.0 : -1.0)
                : (ops.Compare(labels[i], ops.One) == 0 ? 1.0 : -1.0);
            var g = SegmentationTensorOps.GaussianMask<T>(h, w, px, py, sigma);
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                    expected[row, col] = ops.Add(expected[row, col], ops.FromDouble(sign * ops.ToDouble(g[row, col])));
        }

        var actual = SegmentationPostProcessing.PointAttention(points, labels, h, w, sigma, positiveAtOrAbove);

        AssertSame(expected, actual, "point attention");
        // Labels 0.75 and 2 are negative under the exact-one rule and positive under the 0.5 floor
        Assert.Contains(Flat(actual), v => D(v) > 0.5);
        Assert.Contains(Flat(actual), v => D(v) < -0.5);
    }

    [Fact]
    public void PointAttention_ExactOneLabels_MatchesThePerPixelAccumulation_Double() => PointAttentionMatchesScan<double>(null);

    [Fact]
    public void PointAttention_ExactOneLabels_MatchesThePerPixelAccumulation_Float() => PointAttentionMatchesScan<float>(null);

    [Fact]
    public void PointAttention_FloorLabels_MatchesThePerPixelAccumulation_Double() => PointAttentionMatchesScan<double>(0.5);

    [Fact]
    public void PointAttention_TheTwoLabelRulesDiffer()
    {
        var points = new Tensor<double>(new[] { 3.0, 3.0 }, new[] { 1, 2 });
        var labels = new Tensor<double>(new[] { 0.75 }, new[] { 1 });

        var exact = SegmentationPostProcessing.PointAttention(points, labels, 7, 7, 1.0);
        var floor = SegmentationPostProcessing.PointAttention(points, labels, 7, 7, 1.0, positiveAtOrAbove: 0.5);

        Assert.Equal(-1.0, exact[3, 3]);
        Assert.Equal(1.0, floor[3, 3]);
    }

    [Fact]
    public void PointAttention_NoPoints_IsZero()
    {
        var attention = SegmentationPostProcessing.PointAttention(
            new Tensor<double>(new[] { 0, 2 }), new Tensor<double>(new[] { 0 }), 3, 4, 1.0);

        Assert.Equal(new[] { 3, 4 }, attention.Shape.ToArray());
        Assert.All(Flat(attention), v => Assert.Equal(0.0, v));
    }

    // ------------------------------------------------------------------ score maps

    private static void GatedChannelMeanMatchesScan<T>(int seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int c = 7, h = 5, w = 6;
        var features = RandomTensor<T>(new[] { c, h, w }, seed, -3.0, 3.0);
        var gate = RandomTensor<T>(new[] { h, w }, seed + 1, 0.0, 1.0);

        var expected = new Tensor<T>(new[] { h, w });
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double s = 0;
                for (int ch = 0; ch < c; ch++) s += ops.ToDouble(features[ch, y, x]);
                expected[y, x] = ops.FromDouble(s / c * ops.ToDouble(gate[y, x]));
            }
        }

        AssertSame(expected, SegmentationPostProcessing.GatedChannelMean(features, gate), "gated channel mean");
    }

    [Theory]
    [InlineData(61)]
    [InlineData(62)]
    [InlineData(63)]
    public void GatedChannelMean_MatchesTheDoubleAccumulation_Double(int seed) => GatedChannelMeanMatchesScan<double>(seed);

    [Theory]
    [InlineData(64)]
    [InlineData(65)]
    public void GatedChannelMean_MatchesTheDoubleAccumulation_Float(int seed) => GatedChannelMeanMatchesScan<float>(seed);

    [Theory]
    [InlineData(5, 6)]
    [InlineData(3, 4)]
    [InlineData(8, 9)]
    [InlineData(3, 9)]
    [InlineData(8, 2)]
    public void PositiveMaskGate_IsOneWhereTheMaskIsPositiveAndZeroOutsideIt(int maskHeight, int maskWidth)
    {
        const int h = 5, w = 6;
        var mask = RandomTensor<double>(new[] { maskHeight, maskWidth }, 70 + maskHeight * 10 + maskWidth, -1.0, 1.0);
        mask[0, 0] = 0.0;
        mask[1, 1] = double.NaN;

        var expected = new Tensor<double>(new[] { h, w });
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double value = y < mask.Shape[0] && x < mask.Shape[1] ? mask[y, x] : 0;
                expected[y, x] = value > 0 ? 1.0 : 0.0;
            }
        }

        var gate = SegmentationPostProcessing.PositiveMaskGate(mask, h, w);

        AssertSame(expected, gate, "mask gate");
        Assert.Contains(Flat(gate), v => v == 1.0);
        Assert.Contains(Flat(gate), v => v == 0.0);
    }

    // ------------------------------------------------------------------ multi-threshold masks

    private static void PromptMasksMatchScan<T>(int h, int w, int seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var scoreMap = RandomTensor<T>(new[] { h, w }, seed, -3.0, 3.0);

        var probs = SegmentationTensorOps.Sigmoid(scoreMap);
        double[] thresholds = { 0.3, 0.5, 0.7 };
        var masks = new Tensor<T>(new[] { 3, h, w });
        var scores = new double[3];
        var stability = new double[3];
        for (int m = 0; m < 3; m++)
        {
            double area = 0, confSum = 0, areaLo = 0, areaHi = 0;
            double tLo = thresholds[m] - 0.05, tHi = thresholds[m] + 0.05;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    double v = ops.ToDouble(probs[y, x]);
                    if (v >= thresholds[m]) { masks[m, y, x] = ops.FromDouble(1.0); area++; confSum += v; }
                    if (v >= tLo) areaLo++;
                    if (v >= tHi) areaHi++;
                }
            }

            scores[m] = area > 0 ? confSum / area : 0;
            stability[m] = areaLo > 0 ? areaHi / areaLo : 0;
        }

        int lrH = Math.Max(1, h / 4), lrW = Math.Max(1, w / 4);
        var lowRes = new Tensor<T>(new[] { 1, lrH, lrW });
        for (int y = 0; y < lrH; y++)
            for (int x = 0; x < lrW; x++)
                lowRes[0, y, x] = scoreMap[Math.Min(y * 4, h - 1), Math.Min(x * 4, w - 1)];

        var result = SegmentationPostProcessing.PromptMasksFromScores(scoreMap, h, w);

        AssertSame(masks, result.Masks, "masks");
        AssertSame(lowRes, result.LowResLogits, "low-res logits");
        Assert.Equal(stability, result.StabilityScores);
        Assert.Equal(3, result.Scores.Length);
        for (int m = 0; m < 3; m++)
        {
            // The mean is a whole-tensor sum here and a running sum in the scan: same terms, different
            // association, so the last bits may differ
            Assert.True(Math.Abs(scores[m] - result.Scores[m]) <= 1e-13, $"score {m}: expected {scores[m]:R} but was {result.Scores[m]:R}");
        }
    }

    [Theory]
    [InlineData(9, 13, 81)]
    [InlineData(16, 16, 82)]
    [InlineData(3, 2, 83)]
    [InlineData(1, 1, 84)]
    public void PromptMasksFromScores_MatchesThePixelScan_Double(int h, int w, int seed) => PromptMasksMatchScan<double>(h, w, seed);

    [Theory]
    [InlineData(9, 13, 85)]
    [InlineData(8, 5, 86)]
    public void PromptMasksFromScores_MatchesThePixelScan_Float(int h, int w, int seed) => PromptMasksMatchScan<float>(h, w, seed);

    [Fact]
    public void PromptMasksFromScores_AllLowScores_GiveEmptyMasksAndZeroScores()
    {
        var scoreMap = new Tensor<double>(Enumerable.Repeat(-20.0, 12).ToArray(), new[] { 3, 4 });

        var result = SegmentationPostProcessing.PromptMasksFromScores(scoreMap, 3, 4);

        Assert.All(Flat(result.Masks), v => Assert.Equal(0.0, v));
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, result.Scores);
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, result.StabilityScores);
    }

    // ------------------------------------------------------------------ segment everything

    private static void SegmentEveryClassMatchesScan<T>(int classes, int maxResults, int seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var features = BlockLogits<T>(classes, 12, 16, seed);
        var probs = SegmentationTensorOps.SoftmaxAlongClassDim(features);
        var classMap = SegmentationTensorOps.ArgmaxAlongClassDim(features);
        int h = classMap.Shape[0], w = classMap.Shape[1];

        var expectedMasks = new List<Tensor<T>>();
        var expectedScores = new List<double>();
        for (int cls = 0; cls < classes && expectedMasks.Count < maxResults; cls++)
        {
            double area = 0, confSum = 0;
            var mask = new Tensor<T>(new[] { 1, h, w });
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if ((int)ops.ToDouble(classMap[y, x]) == cls)
                    {
                        mask[0, y, x] = ops.FromDouble(1.0);
                        area++;
                        confSum += ops.ToDouble(probs[cls, y, x]);
                    }
                }
            }

            if (area < 4) continue;
            expectedMasks.Add(mask);
            expectedScores.Add(confSum / area);
        }

        var results = SegmentationPostProcessing.SegmentEveryClass(features, probs, maxResults);

        Assert.NotEmpty(expectedMasks);
        Assert.Equal(expectedMasks.Count, results.Count);
        for (int i = 0; i < results.Count; i++)
        {
            AssertSame(expectedMasks[i], results[i].Masks, $"mask {i}");
            Assert.Equal(new[] { expectedScores[i] }, results[i].Scores);
            Assert.Equal(new[] { expectedScores[i] > 0.7 ? 0.95 : expectedScores[i] }, results[i].StabilityScores);
        }
    }

    [Theory]
    [InlineData(6, 100, 91)]
    [InlineData(6, 2, 92)]
    [InlineData(3, 100, 93)]
    public void SegmentEveryClass_MatchesThePixelScan_Double(int classes, int maxResults, int seed)
        => SegmentEveryClassMatchesScan<double>(classes, maxResults, seed);

    [Fact]
    public void SegmentEveryClass_MatchesThePixelScan_Float() => SegmentEveryClassMatchesScan<float>(6, 100, 94);

    [Fact]
    public void SegmentEveryClass_NoClassWithFourPixels_ReturnsOneEmptyMask()
    {
        // 1 x 3 image: no class can win four pixels
        var features = RandomTensor<double>(new[] { 4, 1, 3 }, 95, -1.0, 1.0);

        var results = SegmentationPostProcessing.SegmentEveryClass(features, SegmentationTensorOps.SoftmaxAlongClassDim(features));

        var only = Assert.Single(results);
        Assert.Equal(new[] { 1, 1, 3 }, only.Masks.Shape.ToArray());
        Assert.All(Flat(only.Masks), v => Assert.Equal(0.0, v));
        Assert.Equal(new[] { 0.0 }, only.Scores);
        Assert.Equal(new[] { 0.0 }, only.StabilityScores);
    }
}
