using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// EAST post-processing works on each word's quadrilateral: locality-aware merging, NMS and rescoring decided overlap
/// and averaged scores over the axis-aligned box instead, so neighbouring rotated words were merged or suppressed and
/// background corners diluted their confidence.
/// </summary>
public class EastRotatedWordTests
{
    private sealed class Probe : EAST<double>
    {
        public Probe() : base(new TextDetectionOptions<double> { InputSize = new[] { 64, 64 }, Size = ModelSize.Nano },
            useRotatedBoxes: false)
        {
        }

        public List<AiDotNet.ComputerVision.Detection.TextDetection.TextRegion<double>> Run(
            Tensor<double> score, Tensor<double> geometry, int width, int height, double threshold)
            => PostProcess(new List<Tensor<double>> { score, geometry }, width, height, threshold);
    }

    // Two parallel 45-degree word strips, four pixels thick. Their boxes overlap (IoU 0.25, above the 0.2 NMS
    // threshold) while the strips themselves are disjoint.
    private static readonly (double X, double Y)[] WordA = { (2, 0), (40, 38), (38, 40), (0, 2) };
    private static readonly (double X, double Y)[] WordB = { (22, 0), (40, 18), (38, 20), (20, 2) };

    private static bool Inside((double X, double Y)[] polygon, double x, double y)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
        {
            if ((polygon[i].Y > y) != (polygon[j].Y > y)
                && x < ((polygon[j].X - polygon[i].X) * (y - polygon[i].Y) / (polygon[j].Y - polygon[i].Y)) + polygon[i].X)
                inside = !inside;
        }
        return inside;
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Neighbouring_rotated_words_stay_separate_and_score_by_their_own_area()
    {
        await System.Threading.Tasks.Task.Yield();
        const int cells = 10, size = 40;
        const double scale = (double)size / cells;
        var score = new Tensor<double>(new[] { 1, 1, cells, cells });
        var geometry = new Tensor<double>(new[] { 1, 8, cells, cells });
        int pixelsA = 0, pixelsB = 0;
        for (int h = 0; h < cells; h++)
            for (int w = 0; w < cells; w++)
            {
                double cx = (w + 0.5) * scale, cy = (h + 0.5) * scale;
                var word = Inside(WordA, cx, cy) ? WordA : Inside(WordB, cx, cy) ? WordB : null;
                if (word is null) continue;
                if (word == WordA) pixelsA++; else pixelsB++;
                score[0, 0, h, w] = 1.0;
                for (int k = 0; k < 4; k++)
                {
                    geometry[0, 2 * k, h, w] = (word[k].X - cx) / scale;
                    geometry[0, (2 * k) + 1, h, w] = (word[k].Y - cy) / scale;
                }
            }
        Assert.True(pixelsA > 1 && pixelsB > 1, $"the fixture lit {pixelsA} and {pixelsB} pixels; each word needs several");

        using var east = new Probe();
        var regions = east.Run(score, geometry, size, size, threshold: 0.5);

        Assert.Equal(2, regions.Count);
        foreach (var region in regions)
            Assert.True(Math.Abs(region.Confidence - 1.0) < 1e-9,
                $"a word scored {region.Confidence}: its confidence averaged cells outside its quadrilateral");
    }
}
