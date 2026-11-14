using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Review checks for the shared grounding post-processing: each engine path against the scalar loop it replaced.
/// </summary>
public class VisionLanguageGroundingReviewTests
{
    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Fact]
    public void QuarterBlockQueries_ConfidenceRisesWithTheBlockMagnitude()
    {
        // DINO-X and Grounding DINO 1.5 score a query with 1 / (1 + exp(-rms(block) + 1)) = sigmoid(rms - 1):
        // a silent block scores sigmoid(-1) = 0.2689 and a strong one approaches 1. Three blocks of four values
        // with rms 0, 1 and 3.
        var decoderOut = new Tensor<double>(new[] { 12 }, new Vector<double>(new[]
        {
            0.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 1.0, -1.0,
            3.0, 3.0, -3.0, 3.0
        }));

        GroundingDetections<double>.QuarterBlockQueries(AiDotNetEngine.Current, decoderOut, 3, out _, out var confidences);

        Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), confidences[0], 1e-12);
        Assert.Equal(0.5, confidences[1], 1e-12);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-2.0)), confidences[2], 1e-12);
        Assert.True(confidences[0] < confidences[1] && confidences[1] < confidences[2]);
    }

    [Fact]
    public void QuarterBlockQueries_QueriesPastAShortOutput_ScoreAsSilentBlocks()
    {
        // Two values for four queries: queries 2 and 3 read nothing, so their rms is 0 and they score sigmoid(-1).
        var decoderOut = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 2.0, -4.0 }));

        GroundingDetections<double>.QuarterBlockQueries(AiDotNetEngine.Current, decoderOut, 4, out _, out var confidences);

        Assert.Equal(1.0 / (1.0 + Math.Exp(-1.0)), confidences[0], 1e-12);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-3.0)), confidences[1], 1e-12);
        Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), confidences[2], 1e-12);
        Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), confidences[3], 1e-12);
    }

    [Fact]
    public void QuarterBlockQueries_EmptyOutput_ScoresEveryQueryAsSilent()
    {
        GroundingDetections<double>.QuarterBlockQueries(AiDotNetEngine.Current, new Tensor<double>(new[] { 0 }), 3,
            out var boxes, out var confidences);

        Assert.Equal(new[] { 3, 4 }, boxes.Shape.ToArray());
        for (int q = 0; q < 3; q++)
            Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), confidences[q], 1e-12);
    }

    private static double ScalarIoU(double[] a, double[] b)
    {
        double iw = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]));
        double ih = Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
        double inter = iw * ih;
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        return union > 1e-8 ? inter / union : 0;
    }

    /// <summary>Greedy NMS over the first eligible candidates, visited by descending confidence (stable).</summary>
    private static double[] ScalarSelect(double[][] boxes, double[] confidences, bool[] eligible, int maxDetections,
        double iouThreshold)
    {
        var candidates = new List<int>();
        for (int i = 0; i < boxes.Length && candidates.Count < maxDetections; i++)
            if (eligible[i]) candidates.Add(i);
        var order = candidates.OrderByDescending(i => confidences[i]).ToList();
        var kept = new List<int>();
        foreach (int i in order)
        {
            if (kept.All(k => ScalarIoU(boxes[k], boxes[i]) <= iouThreshold))
                kept.Add(i);
        }

        if (kept.Count == 0)
            return new double[5];
        return kept.SelectMany(i => boxes[i].Concat(new[] { confidences[i] })).ToArray();
    }

    [Theory]
    [InlineData(1, 40, 40, 0.3)]
    [InlineData(2, 60, 25, 0.5)]
    [InlineData(3, 33, 100, 0.1)]
    public void GroundingSelect_MatchesScalarGreedyNmsOnRandomBoxes(int seed, int count, int maxDetections, double iou)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var boxes = new double[count][];
        var confidences = new double[count];
        var eligible = new bool[count];
        for (int i = 0; i < count; i++)
        {
            double x = rng.NextDouble() * 0.6, y = rng.NextDouble() * 0.6;
            boxes[i] = new[] { x, y, x + 0.05 + rng.NextDouble() * 0.35, y + 0.05 + rng.NextDouble() * 0.35 };
            confidences[i] = rng.NextDouble();
            eligible[i] = rng.NextDouble() > 0.2;
        }

        var kept = GroundingDetections<double>.Select(AiDotNetEngine.Current,
            new Tensor<double>(new[] { count, 4 }, new Vector<double>(boxes.SelectMany(b => b).ToArray())),
            new Tensor<double>(new[] { count }, new Vector<double>(confidences)),
            new Tensor<double>(new[] { count }, new Vector<double>(eligible.Select(e => e ? 1.0 : 0.0).ToArray())),
            maxDetections, iou);

        Assert.Equal(ScalarSelect(boxes, confidences, eligible, maxDetections, iou), kept.ToArray());
    }

    [Fact]
    public void GroundingSelect_Half_MatchesScalarGreedyNms()
    {
        // Coordinates and confidences on a 1/64 lattice are exact in Half, so the scalar reference sees the same
        // numbers.
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        const int count = 30;
        var boxes = new double[count][];
        var confidences = new double[count];
        var eligible = new bool[count];
        for (int i = 0; i < count; i++)
        {
            int x = rng.Next(0, 40), y = rng.Next(0, 40);
            boxes[i] = new[] { x / 64.0, y / 64.0, (x + rng.Next(4, 24)) / 64.0, (y + rng.Next(4, 24)) / 64.0 };
            confidences[i] = rng.Next(1, 512) / 512.0;
            eligible[i] = i % 5 != 0;
        }

        var kept = GroundingDetections<Half>.Select(AiDotNetEngine.Current,
            new Tensor<Half>(new[] { count, 4 }, new Vector<Half>(boxes.SelectMany(b => b).Select(v => (Half)v).ToArray())),
            new Tensor<Half>(new[] { count }, new Vector<Half>(confidences.Select(v => (Half)v).ToArray())),
            new Tensor<Half>(new[] { count }, new Vector<Half>(eligible.Select(e => (Half)(e ? 1.0 : 0.0)).ToArray())),
            20, 0.37);

        Assert.Equal(ScalarSelect(boxes, confidences, eligible, 20, 0.37), kept.ToArray().Select(v => (double)v).ToArray());
    }

    [Theory]
    [InlineData(9, 8, 3)]
    [InlineData(6, 3, 3)]
    [InlineData(4, 1, 2)]
    [InlineData(16, 10, 4)]
    public void PatchOffsetBoxes_MatchTheScalarDecoding(int patches, int dim, int gridSize)
    {
        var features = Random(40 + dim, patches, dim);

        var boxes = GroundingDetections<double>.PatchOffsetBoxes(AiDotNetEngine.Current, features, gridSize);

        for (int p = 0; p < patches; p++)
        {
            double centreX = (p % gridSize + 0.5) / gridSize;
            double centreY = (p / gridSize + 0.5) / gridSize;
            double dx = 0, dy = 0, w = 0, h = 0;
            int quarter = Math.Max(1, dim / 4);
            for (int d = 0; d < quarter && d < dim; d++) dx += features[p, d];
            for (int d = quarter; d < 2 * quarter && d < dim; d++) dy += features[p, d];
            for (int d = 2 * quarter; d < 3 * quarter && d < dim; d++) w += features[p, d];
            for (int d = 3 * quarter; d < dim; d++) h += features[p, d];
            dx = Math.Tanh(dx / quarter) * 0.5 / gridSize;
            dy = Math.Tanh(dy / quarter) * 0.5 / gridSize;
            w = 1.0 / (1.0 + Math.Exp(-w / quarter)) * 3.0 / gridSize;
            h = 1.0 / (1.0 + Math.Exp(-h / quarter)) * 3.0 / gridSize;
            double cx = centreX + dx, cy = centreY + dy;

            Assert.Equal(Math.Max(0, cx - w / 2), boxes[p, 0], 1e-12);
            Assert.Equal(Math.Max(0, cy - h / 2), boxes[p, 1], 1e-12);
            Assert.Equal(Math.Min(1, cx + w / 2), boxes[p, 2], 1e-12);
            Assert.Equal(Math.Min(1, cy + h / 2), boxes[p, 3], 1e-12);
        }
    }

    [Fact]
    public void Take_NegativePositionsReadZero_AndRowNormMatchesTheScalarNorm()
    {
        var flat = Random(50, 7);
        int[] positions = { 5, 6, -1, -1, 0, 1, 2, 3 };

        var taken = GroundingDetections<double>.Take(AiDotNetEngine.Current, flat, positions, new[] { 2, 4 });
        var norms = GroundingDetections<double>.RowNorm(AiDotNetEngine.Current, taken);

        for (int r = 0; r < 2; r++)
        {
            double sumSquares = 0;
            for (int c = 0; c < 4; c++)
            {
                int position = positions[r * 4 + c];
                double expected = position < 0 ? 0.0 : flat[position];
                Assert.Equal(expected, taken[r, c]);
                sumSquares += expected * expected;
            }
            Assert.Equal(Math.Sqrt(sumSquares), norms[r], 1e-12);
        }
    }
}
