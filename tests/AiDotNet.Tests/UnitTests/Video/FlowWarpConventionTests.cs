using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// Pins the pixel-to-normalized-coordinate convention <see cref="FlowWarpHelper"/> must use.
/// </summary>
/// <remarks>
/// <para>
/// <c>IEngine.GridSample(input, grid)</c> is a torchvision-default shim, and torchvision defaults to
/// <c>align_corners=false</c>: normalized -1 and +1 sit at the OUTER EDGES of the border pixels, so the
/// mapping is <c>norm = 2 * (p + 0.5) / extent - 1</c>. The intuitive <c>2p / (extent - 1) - 1</c> is the
/// <c>align_corners=true</c> mapping and was what this helper used, which made the sampler decode every
/// coordinate as <c>p * extent / (extent - 1) - 0.5</c>.
/// </para>
/// <para>
/// That is exactly the kind of defect that survives a normal test suite: the warped output has the right
/// shape, is finite, is smooth, and even moves in the right direction, so shape checks, finiteness checks
/// and "did the error go down" checks all pass while every warp is off by half a pixel plus a slow
/// outward stretch. Only an EXACT expected value catches it, which is why these tests use an affine
/// image — bilinear interpolation reproduces an affine function exactly, so the expected value is known
/// in closed form instead of approximated.
/// </para>
/// </remarks>
public class FlowWarpConventionTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>An affine image, sampled exactly by bilinear interpolation.</summary>
    private static Tensor<double> Affine(int h, int w)
    {
        var t = new Tensor<double>(new[] { 1, h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                t[(y * w) + x] = x + (10.0 * y);
        return t;
    }

    private static Tensor<double> ConstantFlow(int h, int w, double dx, double dy)
    {
        var flow = new Tensor<double>(new[] { 2, h, w });
        int plane = h * w;
        for (int p = 0; p < plane; p++)
        {
            flow[p] = dx;
            flow[plane + p] = dy;
        }
        return flow;
    }

    [Fact]
    public void AZeroFlowIsExactlyTheIdentity()
    {
        // The sharpest possible statement of the convention, and the one the old align_corners=true
        // mapping failed: with no displacement at all, every output pixel must be its own input pixel.
        // Under the mismatched mapping this returned a half-pixel-shifted, slightly stretched copy —
        // still smooth, still finite, still the right shape, and wrong everywhere.
        const int h = 16, w = 16;
        var image = Affine(h, w);

        var warped = FlowWarpHelper.Warp(Engine, image, ConstantFlow(h, w, 0.0, 0.0));

        Assert.Equal(image.Shape.ToArray(), warped.Shape.ToArray());
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                Assert.Equal(image[(y * w) + x], warped[(y * w) + x], 6);
    }

    [Fact]
    public void AnIntegerFlowShiftsByExactlyThatManyPixels()
    {
        // Warp is a BACKWARD warp: out[y][x] = src[y + fy][x + fx]. With an affine source the expected
        // value is exact, so a half-pixel error cannot hide inside interpolation slop.
        const int h = 16, w = 16;
        const double fx = 2.0, fy = 3.0;
        var image = Affine(h, w);

        var warped = FlowWarpHelper.Warp(Engine, image, ConstantFlow(h, w, fx, fy));

        // Interior only: the borders sample outside the image, where the padding mode decides the value.
        for (int y = 0; y < h - (int)fy; y++)
        {
            for (int x = 0; x < w - (int)fx; x++)
            {
                double expected = (x + fx) + (10.0 * (y + fy));
                Assert.Equal(expected, warped[(y * w) + x], 6);
            }
        }
    }

    [Fact]
    public void AHalfPixelFlowInterpolatesMidwayBetweenNeighbours()
    {
        // Confirms the mapping is right at SUB-pixel offsets too, not merely aligned on the integer
        // lattice. A scale error of extent/(extent-1) is invisible at p=0 and grows with p, so this also
        // exercises a coordinate far from the origin.
        const int h = 16, w = 16;
        var image = Affine(h, w);

        var warped = FlowWarpHelper.Warp(Engine, image, ConstantFlow(h, w, 0.5, 0.5));

        for (int y = 0; y < h - 1; y++)
        {
            for (int x = 0; x < w - 1; x++)
            {
                double expected = (x + 0.5) + (10.0 * (y + 0.5));
                Assert.Equal(expected, warped[(y * w) + x], 6);
            }
        }
    }
}
