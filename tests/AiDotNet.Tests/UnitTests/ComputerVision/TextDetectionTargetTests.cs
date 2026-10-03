using AiDotNet.ComputerVision.Detection.TextDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// The text detectors' paper target encoders, checked against values worked out by hand from each paper's
/// definition. Each map is one pixel per image pixel, so map coordinates equal image coordinates.
/// </summary>
public class TextDetectionTargetTests
{
    private static TextPolygonTarget[] One(TextPolygonTarget target) => new[] { target };

    [Fact]
    public void Geometry_OffsetMovesEveryEdgeByTheDistance()
    {
        var square = new[] { (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0) };
        var shrunk = TextTargetGeometry.Offset(square, 2.0);
        var grown = TextTargetGeometry.Offset(square, -1.0);
        Assert.NotNull(shrunk);
        Assert.NotNull(grown);
        Assert.Equal(36.0, TextTargetGeometry.Area(shrunk), 9);
        Assert.Equal(144.0, TextTargetGeometry.Area(grown), 9);
        Assert.Null(TextTargetGeometry.Offset(square, 6.0));
    }

    [Fact]
    public void Geometry_MinAreaRectangleOfADiamondIsTheDiamond()
    {
        var diamond = new[] { (5.0, 0.0), (10.0, 5.0), (5.0, 10.0), (0.0, 5.0) };
        Assert.Equal(50.0, TextTargetGeometry.Area(TextTargetGeometry.MinAreaRectangle(diamond)), 9);
    }

    [Fact]
    public void Geometry_InverseQuadMappingSendsTheCornersToTheUnitSquare()
    {
        var quad = new[] { (1.0, 2.0), (9.0, 1.0), (10.0, 7.0), (2.0, 8.0) };
        var inverse = TextTargetGeometry.InverseQuadMapping(quad);
        Assert.NotNull(inverse);
        var expected = new[] { (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) };
        for (int i = 0; i < 4; i++)
        {
            var (u, v) = inverse(quad[i].Item1, quad[i].Item2);
            Assert.Equal(expected[i].Item1, u, 9);
            Assert.Equal(expected[i].Item2, v, 9);
        }
    }

    [Fact]
    public void DBNet_KernelIsShrunkByTheVattiDistance_AndBorderIsOneMinusDistanceOverD()
    {
        // A = 800 and L = 120, so D = A (1 - 0.4^2) / L = 5.6, and the kernel is (5.6, 5.6)..(34.4, 14.4).
        var (kernel, mask, border, borderMask) = DBNet<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(0, 0, 40, 20)), 40, 40, 40, 40);
        Assert.Equal(1.0, kernel[6, 6]);
        Assert.Equal(0.0, kernel[5, 5]);
        // Pixel centres: (33.5, 13.5) is inside, (34.5, 13.5) and (33.5, 14.5) are past the 34.4 / 14.4 edges.
        Assert.Equal(1.0, kernel[13, 33]);
        Assert.Equal(0.0, kernel[13, 34]);
        Assert.Equal(0.0, kernel[14, 33]);
        Assert.Equal(1.0, mask[0, 0]);

        // Pixel centre (20.5, 0.5) is 0.5 from the top edge: 1 - 0.5 / 5.6, mapped into [0.3, 0.7].
        Assert.Equal(((1.0 - (0.5 / 5.6)) * 0.4) + 0.3, border[0, 20], 9);
        Assert.Equal(1.0, borderMask[0, 20]);
        // The middle is farther than D from every edge, so it gets the band's floor and is still supervised.
        Assert.Equal(0.3, border[10, 20], 9);
        Assert.Equal(1.0, borderMask[10, 20]);
        // Outside the dilated polygon: unsupervised.
        Assert.Equal(0.0, borderMask[30, 20]);
    }

    [Fact]
    public void DBNet_PolygonTooSmallToShrinkIsMaskedOut()
    {
        var (kernel, mask, _, _) = DBNet<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(10, 10, 10.5, 10.5)), 40, 40, 40, 40);
        Assert.All(kernel.Cast<double>(), value => Assert.Equal(0.0, value));
        Assert.Equal(1.0, mask[0, 0]);
    }

    [Fact]
    public void EAST_RboxGeometryIsTheDistanceToEachEdge_InsideTheShrunkQuad()
    {
        // Shortest edge 16, so the score region is the box moved in by 0.3 * 16 = 4.8: (4.8, 4.8)..(27.2, 11.2).
        var (score, geometry, norm) = EAST<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(0, 0, 32, 16)), 32, 32, 32, 32, rotatedBoxes: true);
        Assert.Equal(1.0, score[8, 10]);
        Assert.Equal(0.0, score[8, 2]);
        Assert.Equal(0.0, score[2, 10]);
        // Either side of the 4.8 inset: centre x 4.5 is outside it, 5.5 inside.
        Assert.Equal(0.0, score[8, 4]);
        Assert.Equal(1.0, score[8, 5]);

        // Pixel centre (10.5, 8.5): top, right, bottom, left, angle.
        Assert.Equal(8.5, geometry[0][8, 10], 9);
        Assert.Equal(21.5, geometry[1][8, 10], 9);
        Assert.Equal(7.5, geometry[2][8, 10], 9);
        Assert.Equal(10.5, geometry[3][8, 10], 9);
        Assert.Equal(0.0, geometry[4][8, 10], 9);
        Assert.Equal(1.0 / (8.0 * 16.0), norm[8, 10], 12);
        Assert.Equal(0.0, geometry[1][8, 2]);
    }

    [Fact]
    public void EAST_QuadGeometryIsTheOffsetToEachVertex_FromTheTopLeftOne()
    {
        var (_, geometry, _) = EAST<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(0, 0, 32, 16)), 32, 32, 32, 32, rotatedBoxes: false);
        Assert.Equal(8, geometry.Length);
        var expected = new[] { -10.5, -8.5, 21.5, -8.5, 21.5, 7.5, -10.5, 7.5 };
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], geometry[i][8, 10], 9);
    }

    [Theory]
    [InlineData("HI")]
    [InlineData("H I")]
    [InlineData(null)]
    public void CRAFT_SplitsTheWordIntoCharacterGaussians_AndLinksNeighbours(string? transcription)
    {
        // Two 12 x 12 characters (from the transcription, or from the 2:1 aspect ratio without one).
        var (region, affinity) = CRAFT<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(0, 0, 24, 12, transcription)), 24, 12, 24, 12);

        // Each character's Gaussian peaks at its centre: pixel centre (5.5, 5.5) is (0.458, 0.458) in the box.
        double nearCentre = Math.Exp(-2 * Math.Pow((5.5 / 12) - 0.5, 2) / (2 * 0.25 * 0.25));
        Assert.Equal(nearCentre, region[5, 5], 9);
        Assert.Equal(nearCentre, region[5, 17], 9);
        // Between the characters the region map dips.
        Assert.True(region[5, 11] < 0.25, $"region between the characters is {region[5, 11]}");

        // The affinity box joins the triangle centres (6, 2), (18, 2), (18, 10), (6, 10), so it peaks between
        // the characters and is empty over their centres.
        Assert.True(affinity[5, 11] > 0.9, $"affinity between the characters is {affinity[5, 11]}");
        Assert.Equal(0.0, affinity[5, 5]);
    }

    [Fact]
    public void CRAFT_SingleCharacterHasNoAffinity()
    {
        var (region, affinity) = CRAFT<double>.BuildTargets(
            One(TextPolygonTarget.FromBox(0, 0, 12, 12, "H")), 12, 12, 12, 12);
        Assert.True(region[5, 5] > 0.95);
        Assert.All(affinity.Cast<double>(), value => Assert.Equal(0.0, value));
    }
}
