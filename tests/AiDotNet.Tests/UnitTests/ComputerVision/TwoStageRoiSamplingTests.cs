using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// FasterRCNN and MaskRCNN sample the RoIs a stage trains on before their box head, as torchvision's RoIHeads does,
/// instead of running the head on every proposal and sampling inside the loss. A 320x320 FasterRCNN step went from
/// 8-9 s and a 6-7 GB working set to ~2 s and ~2.9 GB. The loss then sees exactly the sample, so it must keep every row
/// of it and draw nothing more: otherwise the head would train on a different set than the one it was run on.
/// </summary>
public class TwoStageRoiSamplingTests
{
    private static readonly double[][] Gold = { new[] { 40.0, 40.0, 120.0, 120.0 }, new[] { 150.0, 60.0, 230.0, 160.0 } };

    [Fact]
    public void Sample_RespectsTheBatchAndForegroundLimits()
    {
        var loss = new TwoStageDetectionLoss<double>(5, 1, new TwoStageDetectionLossOptions());
        var proposals = Proposals(new Random(1), 1000);

        var rows = loss.SampleStageRois(proposals, Gold, 0, new Random(2));

        Assert.InRange(rows.Length, 1, 64);
        Assert.Equal(rows.Length, rows.Distinct().Count());
        int foreground = rows.Count(r => BestIoU(proposals, r) >= 0.5);
        Assert.InRange(foreground, 1, 16);
    }

    [Fact]
    public void Resampling_TheSample_KeepsEveryRowAndDrawsNothing()
    {
        var loss = new TwoStageDetectionLoss<double>(5, 1, new TwoStageDetectionLossOptions());
        var proposals = Proposals(new Random(3), 1000);
        var rows = loss.SampleStageRois(proposals, Gold, 0, new Random(4));
        var sample = Rows(proposals, rows);

        var counting = new CountingRandom(5);
        var again = loss.SampleStageRois(sample, Gold, 0, counting);

        Assert.Equal(Enumerable.Range(0, rows.Length), again.OrderBy(r => r));
        Assert.Equal(0, counting.Draws);
    }

    private static Tensor<double> Proposals(Random random, int count)
    {
        var boxes = new Tensor<double>([count, 4]);
        for (int i = 0; i < count; i++)
        {
            double cx = random.NextDouble() * 320, cy = random.NextDouble() * 320;
            double w = 10 + random.NextDouble() * 120, h = 10 + random.NextDouble() * 120;
            boxes[i, 0] = cx - w / 2; boxes[i, 1] = cy - h / 2; boxes[i, 2] = cx + w / 2; boxes[i, 3] = cy + h / 2;
        }

        // The object boxes themselves, as the detectors append them.
        for (int g = 0; g < Gold.Length; g++)
            for (int k = 0; k < 4; k++) boxes[g, k] = Gold[g][k];
        return boxes;
    }

    private static Tensor<double> Rows(Tensor<double> boxes, int[] rows)
    {
        var result = new Tensor<double>([rows.Length, 4]);
        for (int i = 0; i < rows.Length; i++)
            for (int k = 0; k < 4; k++) result[i, k] = boxes[rows[i], k];
        return result;
    }

    private static double BestIoU(Tensor<double> boxes, int row)
    {
        double best = 0;
        foreach (var g in Gold)
        {
            double ix = Math.Max(0, Math.Min(boxes[row, 2], g[2]) - Math.Max(boxes[row, 0], g[0]));
            double iy = Math.Max(0, Math.Min(boxes[row, 3], g[3]) - Math.Max(boxes[row, 1], g[1]));
            double inter = ix * iy;
            double union = (boxes[row, 2] - boxes[row, 0]) * (boxes[row, 3] - boxes[row, 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter;
            best = Math.Max(best, union > 0 ? inter / union : 0);
        }

        return best;
    }

    private sealed class CountingRandom : Random
    {
        public CountingRandom(int seed) : base(seed) { }
        public int Draws { get; private set; }
        public override int Next(int maxValue) { Draws++; return base.Next(maxValue); }
        public override int Next(int minValue, int maxValue) { Draws++; return base.Next(minValue, maxValue); }
        public override int Next() { Draws++; return base.Next(); }
        public override double NextDouble() { Draws++; return base.NextDouble(); }
    }
}
