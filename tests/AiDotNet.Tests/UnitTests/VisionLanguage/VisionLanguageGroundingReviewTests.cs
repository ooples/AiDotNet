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

}
