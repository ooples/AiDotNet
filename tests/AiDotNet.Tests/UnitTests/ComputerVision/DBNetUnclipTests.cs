using AiDotNet.ComputerVision.Detection.TextDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// DBNet's inference dilation, D' = A' r' / L' with r' = 1.5 (Liao et al. 2020, Eq. 10), checked against
/// values worked by hand from the formula rather than from the code.
/// </summary>
public sealed class DBNetUnclipTests
{
    [Fact]
    public void AxisAlignedKernel_GrowsByTheFormulaOffsetOnEverySide()
    {
        // Boundary pixels of a 10 x 4 kernel: A' = 40, L' = 28, D' = 40 * 1.5 / 28 = 2.142857...
        var points = new List<(int H, int W)>();
        for (int w = 0; w <= 10; w++) { points.Add((0, w)); points.Add((4, w)); }
        for (int h = 0; h <= 4; h++) { points.Add((h, 0)); points.Add((h, 10)); }

        var box = DBNet<double>.UnclipMinAreaRectangle(points, DBNet<double>.UnclipRatio);

        double d = 40 * 1.5 / 28;
        Assert.Equal(4, box.Count);
        Assert.Equal(-d, box.Min(p => p.X), 9);
        Assert.Equal(10 + d, box.Max(p => p.X), 9);
        Assert.Equal(-d, box.Min(p => p.Y), 9);
        Assert.Equal(4 + d, box.Max(p => p.Y), 9);
    }

    [Fact]
    public void RotatedKernel_KeepsItsOrientation()
    {
        // A square rotated 45 degrees (a diamond of half-diagonal 5): side 5*sqrt(2), so the minimum-area
        // rectangle is that square, not the 10 x 10 axis box. Each side grows by 2 D', D' = side * 1.5 / 4.
        var points = new List<(int H, int W)> { (0, 5), (5, 10), (10, 5), (5, 0) };

        var box = DBNet<double>.UnclipMinAreaRectangle(points, DBNet<double>.UnclipRatio);

        double side = 5 * Math.Sqrt(2);
        double grown = side + 2 * (side * side * 1.5 / (4 * side));
        for (int i = 0; i < 4; i++)
        {
            var a = box[i];
            var b = box[(i + 1) % 4];
            Assert.Equal(grown, Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y)), 9);
        }
    }
}