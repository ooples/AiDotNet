using System;
using System.Collections.Generic;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision.OCR;

/// <summary>
/// Verifies ABCNet's two contributions — the cubic Bezier text representation and BezierAlign
/// (Liu et al., CVPR 2020, arXiv:2002.10200).
/// </summary>
/// <remarks>
/// Both are parameter-free geometry, so their correctness is a property of the operation rather than of
/// trained weights, which makes them fully checkable. Each test targets a specific way the scheme gets
/// silently simplified: uniform instead of chord-length parameterization, pinned instead of fitted end
/// control points, an axis-aligned instead of curve-following sampling grid, pixel corners instead of
/// centres, and a swapped top/bottom curve.
/// </remarks>
public class BezierTextRepresentationTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static List<(double X, double Y)> SampleCurve(
        CubicBezierCurve<double> curve, int count)
    {
        var pts = new List<(double X, double Y)>(count);
        for (int i = 0; i < count; i++) pts.Add(curve.Evaluate(i / (double)(count - 1)));
        return pts;
    }

    // ---------------- the Bezier representation ----------------

    [Fact]
    public void TheBernsteinBasisIsAPartitionOfUnity()
    {
        // The basis summing to 1 at every t is what makes the curve a convex combination of its control
        // points. A dropped factor of 3 on either middle term — the single easiest slip in writing the
        // cubic out — breaks this and leaves a curve that still looks like a curve.
        foreach (double t in new[] { 0.0, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0 })
        {
            var b = CubicBezierCurve<double>.BernsteinBasis(t);
            Assert.Equal(4, b.Length);
            Assert.Equal(1.0, b[0] + b[1] + b[2] + b[3], 12);
            foreach (double v in b) Assert.True(v >= -1e-12, $"basis value {v:R} is negative at t={t:R}.");
        }
    }

    [Fact]
    public void TheCurvePassesThroughItsEndControlPoints()
    {
        // A cubic Bezier interpolates P0 and P3 but only approximates P1 and P2. Asserting the ends is
        // how an off-by-one in the basis ordering (which reverses the curve) gets caught.
        var curve = new CubicBezierCurve<double>(new[] { (1.0, 2.0), (4.0, 9.0), (11.0, 9.0), (14.0, 2.0) });

        var start = curve.Evaluate(0.0);
        var end = curve.Evaluate(1.0);

        Assert.Equal(1.0, start.X, 12);
        Assert.Equal(2.0, start.Y, 12);
        Assert.Equal(14.0, end.X, 12);
        Assert.Equal(2.0, end.Y, 12);
    }

    /// <summary>Smallest distance from a point to the fitted curve, sampled densely.</summary>
    private static double DistanceToCurve(CubicBezierCurve<double> curve, (double X, double Y) p)
    {
        double best = double.MaxValue;
        for (int i = 0; i <= 400; i++)
        {
            var (cx, cy) = curve.Evaluate(i / 400.0);
            double d = Math.Sqrt(((cx - p.X) * (cx - p.X)) + ((cy - p.Y) * (cy - p.Y)));
            if (d < best) best = d;
        }
        return best;
    }

    [Fact]
    public void TheFittedCurvePassesCloseToEveryAnnotatedPoint()
    {
        // NOTE ON WHAT IS *NOT* ASSERTED, because the obvious stronger claim is false. Sampling a known
        // cubic at uniform t and fitting must NOT be expected to return the original control points: the
        // fit derives each point's parameter from cumulative CHORD LENGTH, and for a curved cubic arc
        // length is not proportional to t, so the recovered parameters legitimately differ from the ones
        // the samples were generated at. An earlier version of this test demanded exact recovery and
        // failed by ~2%, which was the test being wrong rather than the fit.
        //
        // What least squares does guarantee, and what ABCNet actually needs, is that the curve passes
        // close to the annotated boundary points. That is what is measured here.
        var truth = new CubicBezierCurve<double>(new[] { (2.0, 3.0), (6.0, 12.0), (15.0, 11.0), (20.0, 4.0) });
        var points = SampleCurve(truth, 24);
        var fitted = CubicBezierCurve<double>.Fit(points);

        // Span of the data, so the tolerance is relative to the instance's size rather than absolute.
        double span = Math.Sqrt((18.0 * 18.0) + (9.0 * 9.0));
        double worst = 0.0;
        foreach (var p in points) worst = Math.Max(worst, DistanceToCurve(fitted, p));

        Assert.True(worst < 0.02 * span,
            $"The fitted curve missed an annotated point by {worst:F4} against a {span:F2} span "
            + $"({worst / span:P2}). A working least-squares fit stays far closer than this.");
    }

    [Fact]
    public void FittingIsInsensitiveToHowDenselyTheEdgeWasAnnotated()
    {
        // THE discriminating test for chord-length parameterization, and it needs no access to internals.
        // Chord length is a property of the GEOMETRY, so re-annotating the same curve with a different
        // density of vertices must yield essentially the same fitted curve. Under uniform index spacing
        // it would not: crowding two thirds of the vertices into the first quarter would drag the fitted
        // curve toward that stretch, which is exactly how real annotations (dense where text bends,
        // sparse where it is straight) get distorted.
        var truth = new CubicBezierCurve<double>(new[] { (0.0, 0.0), (2.0, 14.0), (18.0, 14.0), (20.0, 0.0) });

        var even = CubicBezierCurve<double>.Fit(SampleCurve(truth, 40));

        var uneven = new List<(double X, double Y)>();
        for (int i = 0; i < 16; i++) uneven.Add(truth.Evaluate(0.25 * (i / 15.0)));
        for (int i = 1; i <= 8; i++) uneven.Add(truth.Evaluate(0.25 + (0.75 * (i / 8.0))));
        var crowded = CubicBezierCurve<double>.Fit(uneven);

        double span = 20.0;
        double worst = 0.0;
        for (int i = 0; i <= 100; i++)
        {
            double t = i / 100.0;
            var (ex, ey) = even.Evaluate(t);
            var (ax, ay) = crowded.Evaluate(t);
            worst = Math.Max(worst, Math.Sqrt(((ex - ax) * (ex - ax)) + ((ey - ay) * (ey - ay))));
        }

        Assert.True(worst < 0.05 * span,
            $"Re-annotating the same curve with a different vertex density moved the fitted curve by "
            + $"{worst:F4} against a {span:F1} span ({worst / span:P1}). The fit is being driven by vertex "
            + "COUNT rather than by geometry, so it is not parameterizing by chord length.");
    }

    [Fact]
    public void FittingAStraightEdgeGivesAStraightCurve()
    {
        // A rectangular text instance is the common case and produces collinear points. The fit must not
        // introduce curvature, and it must not divide by zero on the rank-deficient two-point input.
        var fitted = CubicBezierCurve<double>.Fit(new[] { (3.0, 5.0), (13.0, 5.0) });

        for (int i = 0; i <= 10; i++)
        {
            double t = i / 10.0;
            var (x, y) = fitted.Evaluate(t);
            Assert.Equal(5.0, y, 6);
            Assert.Equal(3.0 + (10.0 * t), x, 6);
        }
    }

    [Fact]
    public void FittingRejectsInputThatCannotDetermineACurve()
    {
        Assert.Throws<ArgumentException>(() => CubicBezierCurve<double>.Fit(new[] { (1.0, 1.0) }));
        Assert.Throws<ArgumentNullException>(() => CubicBezierCurve<double>.Fit(null!));
        Assert.Throws<ArgumentException>(() =>
            new CubicBezierCurve<double>(new[] { (0.0, 0.0), (1.0, 1.0), (2.0, 2.0) }));
    }

    // ---------------- BezierAlign ----------------

    /// <summary>An AFFINE feature map. Bilinear interpolation reproduces an affine function exactly, so
    /// the expected sampled values are known in closed form rather than approximated.</summary>
    private static Tensor<double> AffineFeatures(int h, int w)
    {
        var t = new Tensor<double>(new[] { 1, h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                t[(y * w) + x] = x + (10.0 * y);
        return t;
    }

    private static CubicBezierCurve<double> StraightEdge(double x0, double x1, double y)
    {
        // Equally spaced control points along a line make B(t) exactly linear in t, so the geometry
        // under test stays in closed form.
        double step = (x1 - x0) / 3.0;
        return new CubicBezierCurve<double>(new[]
        {
            (x0, y), (x0 + step, y), (x0 + (2 * step), y), (x1, y),
        });
    }

    [Fact]
    public void EverySamplingCoefficientRowSumsToOne()
    {
        // The stated invariant, and the cheapest way to catch a sign slip or a mis-indexed half of the
        // top/bottom blend: the row is a convex combination of the eight control points.
        var coeff = BezierAlign.SamplingCoefficients<double>(outputHeight: 4, outputWidth: 7);

        Assert.Equal(new[] { 28, 8 }, coeff.Shape.ToArray());
        for (int row = 0; row < 28; row++)
        {
            double sum = 0.0;
            for (int k = 0; k < 8; k++) sum += coeff[(row * 8) + k];
            Assert.Equal(1.0, sum, 12);
        }
    }

    [Fact]
    public void AStraightInstanceRectifiesToTheExpectedResampledCrop()
    {
        // Ground truth in closed form. With both edges straight, the sampling location is
        // x = 2 + 10t and y = 4 + 6s for t = (j+0.5)/outW and s = (i+0.5)/outH, and the affine feature
        // map makes the bilinear result exact. This pins the HALF-PIXEL CENTRE convention: sampling at
        // j/outW instead of (j+0.5)/outW shifts every value and fails here.
        const int h = 16, w = 16, outH = 3, outW = 5;
        var features = AffineFeatures(h, w);

        var top = StraightEdge(2.0, 12.0, 4.0);
        var bottom = StraightEdge(2.0, 12.0, 10.0);
        var cp = BezierAlign.ControlPointTensor(top, bottom);

        var rect = BezierAlign.Sample(Engine, features, cp, outH, outW);

        Assert.Equal(new[] { 1, outH, outW }, rect.Shape.ToArray());
        for (int i = 0; i < outH; i++)
        {
            double s = (i + 0.5) / outH;
            for (int j = 0; j < outW; j++)
            {
                double t = (j + 0.5) / outW;
                double expected = (2.0 + (10.0 * t)) + (10.0 * (4.0 + (6.0 * s)));
                Assert.Equal(expected, rect[(i * outW) + j], 6);
            }
        }
    }

    [Fact]
    public void ACurvedInstanceIsSampledAlongItsCurveNotAnAxisAlignedBox()
    {
        // The point of BezierAlign. A bright ridge is painted along a curved path; the two boundary
        // curves bracket it. Sampling along the curve keeps the ridge on one rectified row everywhere.
        // An axis-aligned RoIAlign box over the same extent cannot — the ridge would wander across rows
        // and the middle row's minimum would collapse.
        const int h = 40, w = 40, outH = 5, outW = 16;

        // Ridge centre: y = 20 + 8*sin(pi*u) as x goes 4 -> 36.
        static double RidgeY(double u) => 20.0 + (8.0 * Math.Sin(Math.PI * u));

        var features = new Tensor<double>(new[] { 1, h, w });
        for (int x = 4; x <= 36; x++)
        {
            double u = (x - 4) / 32.0;
            int y = (int)Math.Round(RidgeY(u));
            for (int dy = -2; dy <= 2; dy++)
            {
                int yy = y + dy;
                if (yy >= 0 && yy < h) features[(yy * w) + x] = 1.0;
            }
        }

        // Fit the two boundary curves to the ridge's own edges, so they genuinely follow it.
        var topPts = new List<(double X, double Y)>();
        var botPts = new List<(double X, double Y)>();
        for (int k = 0; k <= 16; k++)
        {
            double u = k / 16.0;
            double x = 4.0 + (32.0 * u);
            topPts.Add((x, RidgeY(u) - 2.0));
            botPts.Add((x, RidgeY(u) + 2.0));
        }

        var cp = BezierAlign.ControlPointTensor(
            CubicBezierCurve<double>.Fit(topPts), CubicBezierCurve<double>.Fit(botPts));

        var rect = BezierAlign.Sample(Engine, features, cp, outH, outW);

        // Every column of the middle row must land inside the ridge.
        int mid = outH / 2;
        double worstMiddle = double.MaxValue;
        for (int j = 0; j < outW; j++) worstMiddle = Math.Min(worstMiddle, rect[(mid * outW) + j]);

        Assert.True(worstMiddle > 0.9,
            $"The rectified middle row dropped to {worstMiddle:F3} somewhere along the instance, so the "
            + "sampling grid is not following the curve — it is drifting off the text.");
    }

    [Fact]
    public void SwappingTheTopAndBottomCurvesFlipsTheRectifiedOutput()
    {
        // Documents that curve ORDER is load-bearing, and proves the two halves of the blend are not
        // accidentally symmetric — a bug that would make the vertical direction meaningless while still
        // producing a plausible feature map.
        const int h = 16, w = 16, outH = 4, outW = 4;
        var features = AffineFeatures(h, w);

        var top = StraightEdge(2.0, 12.0, 4.0);
        var bottom = StraightEdge(2.0, 12.0, 10.0);

        var normal = BezierAlign.Sample(Engine, features, BezierAlign.ControlPointTensor(top, bottom), outH, outW);
        var swapped = BezierAlign.Sample(Engine, features, BezierAlign.ControlPointTensor(bottom, top), outH, outW);

        for (int i = 0; i < outH; i++)
            for (int j = 0; j < outW; j++)
                Assert.Equal(normal[(i * outW) + j], swapped[(((outH - 1 - i) * outW) + j)], 6);
    }

    [Fact]
    public void SampleRejectsShapesItCannotInterpret()
    {
        var features = AffineFeatures(8, 8);
        var top = StraightEdge(1.0, 6.0, 2.0);
        var cp = BezierAlign.ControlPointTensor(top, StraightEdge(1.0, 6.0, 6.0));

        Assert.Throws<ArgumentNullException>(() => BezierAlign.Sample<double>(null!, features, cp, 2, 2));
        Assert.Throws<ArgumentException>(() =>
            BezierAlign.Sample(Engine, features, new Tensor<double>(new[] { 4, 2 }), 2, 2));

        // A multi-image batch cannot be described by one instance's control points.
        Assert.Throws<ArgumentException>(() =>
            BezierAlign.Sample(Engine, new Tensor<double>(new[] { 2, 1, 8, 8 }), cp, 2, 2));
    }
}
