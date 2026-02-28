using AiDotNet.Interpolation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Interpolation;

public class InterpolationDeepMathIntegrationTests
{
    private const double Tol = 1e-10;
    private const double ModerateTol = 1e-6;

    // Helper: create Vector<double> from array
    private static Vector<double> V(params double[] vals) => new(vals);

    // ──────────────────────────────────────────────────────────
    // LINEAR INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Linear_ExactAtKnots()
    {
        // Linear interpolation must reproduce the data exactly at knot points
        var x = V(1, 2, 3, 4, 5);
        var y = V(10, 20, 30, 40, 50);
        var interp = new LinearInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void Linear_MidpointHandValues()
    {
        // Between (1,10) and (3,30), midpoint x=2 should give y=20
        var x = V(1, 3, 5);
        var y = V(10, 30, 50);
        var interp = new LinearInterpolation<double>(x, y);

        Assert.Equal(20.0, interp.Interpolate(2.0), Tol);
        Assert.Equal(40.0, interp.Interpolate(4.0), Tol);
    }

    [Fact]
    public void Linear_ReproducesLinearFunction()
    {
        // f(x) = 3x + 7 should be reproduced exactly by linear interpolation
        var x = V(0, 1, 2, 3, 4, 5);
        var y = V(7, 10, 13, 16, 19, 22);
        var interp = new LinearInterpolation<double>(x, y);

        // Check at non-knot points
        Assert.Equal(3 * 0.5 + 7, interp.Interpolate(0.5), Tol);
        Assert.Equal(3 * 1.7 + 7, interp.Interpolate(1.7), Tol);
        Assert.Equal(3 * 3.33 + 7, interp.Interpolate(3.33), Tol);
    }

    [Fact]
    public void Linear_MonotonicallyIncreasingData()
    {
        // If data is monotonically increasing, interpolation should also be
        var x = V(0, 1, 2, 3, 4);
        var y = V(1, 3, 5, 8, 12);
        var interp = new LinearInterpolation<double>(x, y);

        double prev = interp.Interpolate(0.0);
        for (double t = 0.1; t <= 4.0; t += 0.1)
        {
            double curr = interp.Interpolate(t);
            Assert.True(curr >= prev - Tol, $"Not monotone at t={t}: {curr} < {prev}");
            prev = curr;
        }
    }

    [Fact]
    public void Linear_SymmetricData()
    {
        // Symmetric data about x=2: f(2-d) = f(2+d)
        var x = V(0, 1, 2, 3, 4);
        var y = V(10, 5, 0, 5, 10);
        var interp = new LinearInterpolation<double>(x, y);

        Assert.Equal(interp.Interpolate(0.5), interp.Interpolate(3.5), Tol);
        Assert.Equal(interp.Interpolate(1.2), interp.Interpolate(2.8), Tol);
    }

    [Fact]
    public void Linear_QuarterPoint()
    {
        // At x=1.25 between (1,10) and (2,20): y = 10 + 0.25*(20-10) = 12.5
        var x = V(1, 2, 3);
        var y = V(10, 20, 30);
        var interp = new LinearInterpolation<double>(x, y);

        Assert.Equal(12.5, interp.Interpolate(1.25), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // CUBIC SPLINE INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void CubicSpline_ExactAtKnots()
    {
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 1, 0, 1, 0);
        var interp = new CubicSplineInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void CubicSpline_ReproducesLinearFunction()
    {
        // A natural cubic spline through linear data should reproduce the line exactly
        // (all second derivatives are zero, matching natural BC)
        var x = V(0, 1, 2, 3, 4, 5);
        var y = V(2, 5, 8, 11, 14, 17);
        var interp = new CubicSplineInterpolation<double>(x, y);

        Assert.Equal(3 * 0.5 + 2, interp.Interpolate(0.5), Tol);
        Assert.Equal(3 * 2.7 + 2, interp.Interpolate(2.7), Tol);
        Assert.Equal(3 * 4.1 + 2, interp.Interpolate(4.1), Tol);
    }

    [Fact]
    public void CubicSpline_SmootherThanLinear()
    {
        // Cubic spline should produce smoother transitions than linear interpolation
        // at the same query point between knots
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 1, 0, 1, 0);
        var cubic = new CubicSplineInterpolation<double>(x, y);
        var linear = new LinearInterpolation<double>(x, y);

        // The cubic spline value near a knot should differ from linear
        // (cubic adds curvature, while linear just draws straight lines)
        double cubicVal = cubic.Interpolate(0.5);
        double linearVal = linear.Interpolate(0.5);
        // Both should be in [0,1], but they shouldn't be identical for this non-linear data
        Assert.NotEqual(cubicVal, linearVal, 1e-4);
    }

    [Fact]
    public void CubicSpline_QuadraticDataPreserved()
    {
        // f(x) = x^2: Natural cubic spline (c[0]=c[n]=0) does NOT exactly reproduce
        // quadratics since f''(x)=2 != 0 at boundaries. But interior points should be close.
        // Use many points and check at interior, away from boundary BC effects.
        double[] xs = { 0, 0.5, 1, 1.5, 2, 2.5, 3 };
        var x = V(xs);
        var y = V(xs.Select(v => v * v).ToArray());
        var interp = new CubicSplineInterpolation<double>(x, y);

        // Test at interior points away from boundaries (BC effect decays inward)
        Assert.Equal(1.25 * 1.25, interp.Interpolate(1.25), 0.01);
        Assert.Equal(1.75 * 1.75, interp.Interpolate(1.75), 0.01);
    }

    [Fact]
    public void CubicSpline_ContinuityBetweenSegments()
    {
        // The spline should be continuous: approaching a knot from left and right gives the same value
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 2, 1, 3, 0);
        var interp = new CubicSplineInterpolation<double>(x, y);

        double eps = 1e-8;
        for (int i = 1; i < x.Length - 1; i++)
        {
            double fromLeft = interp.Interpolate(x[i] - eps);
            double fromRight = interp.Interpolate(x[i] + eps);
            Assert.Equal(fromLeft, fromRight, 1e-5);
        }
    }

    // ──────────────────────────────────────────────────────────
    // LAGRANGE POLYNOMIAL INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Lagrange_ExactAtKnots()
    {
        var x = V(1, 2, 3, 4);
        var y = V(1, 4, 9, 16);
        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void Lagrange_ReproducesQuadratic()
    {
        // 3 points uniquely define a quadratic. f(x) = x^2
        // Points: (0,0), (1,1), (2,4)
        var x = V(0, 1, 2);
        var y = V(0, 1, 4);
        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        // Check at non-knot points
        Assert.Equal(0.25, interp.Interpolate(0.5), Tol);     // 0.5^2
        Assert.Equal(2.25, interp.Interpolate(1.5), Tol);     // 1.5^2
    }

    [Fact]
    public void Lagrange_ReproducesCubic()
    {
        // 4 points uniquely define a cubic. f(x) = x^3 - 2x + 1
        // Points: (-1,2), (0,1), (1,0), (2,5)
        var x = V(-1, 0, 1, 2);
        var y = V(2, 1, 0, 5); // (-1)^3 - 2(-1)+1 = -1+2+1=2, 0^3-0+1=1, 1-2+1=0, 8-4+1=5

        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        // f(0.5) = 0.125 - 1 + 1 = 0.125
        Assert.Equal(0.125, interp.Interpolate(0.5), Tol);
        // f(1.5) = 3.375 - 3 + 1 = 1.375
        Assert.Equal(1.375, interp.Interpolate(1.5), Tol);
    }

    [Fact]
    public void Lagrange_TwoPointsIsLinear()
    {
        // With 2 points, Lagrange should give linear interpolation
        var x = V(0, 2);
        var y = V(3, 7);
        var lagrange = new LagrangePolynomialInterpolation<double>(x, y);
        var linear = new LinearInterpolation<double>(x, y);

        for (double t = 0; t <= 2; t += 0.2)
            Assert.Equal(linear.Interpolate(t), lagrange.Interpolate(t), Tol);
    }

    [Fact]
    public void Lagrange_SymmetricPolynomial()
    {
        // f(x) = x^2 is symmetric about x=0, so interp(-a) = interp(a)
        var x = V(-2, -1, 0, 1, 2);
        var y = V(4, 1, 0, 1, 4);
        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        Assert.Equal(interp.Interpolate(-0.5), interp.Interpolate(0.5), Tol);
        Assert.Equal(interp.Interpolate(-1.5), interp.Interpolate(1.5), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // NEWTON DIVIDED DIFFERENCE INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Newton_ExactAtKnots()
    {
        var x = V(1, 2, 4, 5);
        var y = V(1, 8, 64, 125);
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void Newton_EquivalentToLagrange()
    {
        // Newton and Lagrange produce the same unique polynomial for the same data
        var x = V(0, 1, 2, 3);
        var y = V(1, 3, 7, 13);
        var newton = new NewtonDividedDifferenceInterpolation<double>(x, y);
        var lagrange = new LagrangePolynomialInterpolation<double>(x, y);

        for (double t = 0; t <= 3; t += 0.3)
            Assert.Equal(lagrange.Interpolate(t), newton.Interpolate(t), Tol);
    }

    [Fact]
    public void Newton_ReproducesQuadratic()
    {
        // f(x) = 2x^2 + 3x + 1: three points uniquely define this
        var x = V(0, 1, 2);
        var y = V(1, 6, 15); // 2*0+0+1=1, 2+3+1=6, 8+6+1=15
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);

        double expected = 2 * 1.5 * 1.5 + 3 * 1.5 + 1; // 4.5+4.5+1 = 10
        Assert.Equal(expected, interp.Interpolate(1.5), Tol);
    }

    [Fact]
    public void Newton_DividedDifferences_LinearDataGivesConstantFirstDiff()
    {
        // For f(x) = 5x + 2, all divided differences beyond order 1 are zero.
        // The polynomial should exactly reproduce the line.
        var x = V(0, 2, 4, 6);
        var y = V(2, 12, 22, 32);
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);

        Assert.Equal(5 * 3.0 + 2, interp.Interpolate(3.0), Tol);
        Assert.Equal(5 * 5.0 + 2, interp.Interpolate(5.0), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // NEAREST NEIGHBOR INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void NearestNeighbor_ExactAtKnots()
    {
        var x = V(1, 2, 3, 4);
        var y = V(10, 20, 30, 40);
        var interp = new NearestNeighborInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void NearestNeighbor_CloserToLeft()
    {
        // x=1.3 is closer to 1 than to 2
        var x = V(1, 2, 3);
        var y = V(10, 20, 30);
        var interp = new NearestNeighborInterpolation<double>(x, y);

        Assert.Equal(10.0, interp.Interpolate(1.3), Tol);
    }

    [Fact]
    public void NearestNeighbor_CloserToRight()
    {
        // x=1.7 is closer to 2 than to 1
        var x = V(1, 2, 3);
        var y = V(10, 20, 30);
        var interp = new NearestNeighborInterpolation<double>(x, y);

        Assert.Equal(20.0, interp.Interpolate(1.7), Tol);
    }

    [Fact]
    public void NearestNeighbor_PiecewiseConstant()
    {
        // NN interpolation should produce a step function
        // All queries between (1, 1.5) should map to y=10 (nearest = x=1)
        // All queries in (1.5, 2.5) should map to y=20 (nearest = x=2)
        var x = V(1, 2, 3);
        var y = V(10, 20, 30);
        var interp = new NearestNeighborInterpolation<double>(x, y);

        Assert.Equal(10.0, interp.Interpolate(1.1), Tol);
        Assert.Equal(10.0, interp.Interpolate(1.4), Tol);
        Assert.Equal(20.0, interp.Interpolate(1.6), Tol);
        Assert.Equal(20.0, interp.Interpolate(2.4), Tol);
        Assert.Equal(30.0, interp.Interpolate(2.6), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // HERMITE INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Hermite_ExactAtKnots()
    {
        var x = V(0, 1, 2);
        var y = V(0, 1, 0);
        var m = V(1, 0, -1); // slopes at each point
        var interp = new HermiteInterpolation<double>(x, y, m);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void Hermite_LinearWithZeroSlope()
    {
        // f(x) = 5 (constant), slopes all zero
        var x = V(0, 1, 2, 3);
        var y = V(5, 5, 5, 5);
        var m = V(0, 0, 0, 0);
        var interp = new HermiteInterpolation<double>(x, y, m);

        Assert.Equal(5.0, interp.Interpolate(0.5), Tol);
        Assert.Equal(5.0, interp.Interpolate(1.7), Tol);
    }

    [Fact]
    public void Hermite_ReproducesLinearWithCorrectSlopes()
    {
        // f(x) = 2x + 3, slope = 2 everywhere
        var x = V(0, 1, 2, 3);
        var y = V(3, 5, 7, 9);
        var m = V(2, 2, 2, 2);
        var interp = new HermiteInterpolation<double>(x, y, m);

        Assert.Equal(2 * 0.5 + 3, interp.Interpolate(0.5), Tol);
        Assert.Equal(2 * 1.3 + 3, interp.Interpolate(1.3), Tol);
        Assert.Equal(2 * 2.8 + 3, interp.Interpolate(2.8), Tol);
    }

    [Fact]
    public void Hermite_ReproducesQuadraticWithCorrectSlopes()
    {
        // f(x) = x^2, slope = 2x
        var x = V(0, 1, 2, 3);
        var y = V(0, 1, 4, 9);
        var m = V(0, 2, 4, 6);
        var interp = new HermiteInterpolation<double>(x, y, m);

        Assert.Equal(0.5 * 0.5, interp.Interpolate(0.5), Tol);
        Assert.Equal(1.5 * 1.5, interp.Interpolate(1.5), Tol);
        Assert.Equal(2.5 * 2.5, interp.Interpolate(2.5), Tol);
    }

    [Fact]
    public void Hermite_CubicExactReproduction()
    {
        // f(x) = x^3, slope = 3x^2
        // Hermite with cubic data on two-knot intervals should be exact
        var x = V(0, 1, 2);
        var y = V(0, 1, 8);
        var m = V(0, 3, 12);
        var interp = new HermiteInterpolation<double>(x, y, m);

        // Cubic Hermite reproduces cubics exactly
        Assert.Equal(0.5 * 0.5 * 0.5, interp.Interpolate(0.5), Tol);
        Assert.Equal(1.5 * 1.5 * 1.5, interp.Interpolate(1.5), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // MONOTONE CUBIC INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void MonotoneCubic_ExactAtKnots()
    {
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 2, 3, 5, 8);
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void MonotoneCubic_PreservesMonotonicity()
    {
        // Monotonically increasing data must have monotonically increasing interpolation
        var x = V(0, 1, 2, 3, 4, 5);
        var y = V(0, 1, 3, 5, 8, 13);
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        double prev = interp.Interpolate(0.0);
        for (double t = 0.05; t <= 5.0; t += 0.05)
        {
            double curr = interp.Interpolate(t);
            Assert.True(curr >= prev - Tol,
                $"Monotonicity violated at t={t:F3}: {curr} < {prev}");
            prev = curr;
        }
    }

    [Fact]
    public void MonotoneCubic_DecreasingDataStaysDecreasing()
    {
        var x = V(0, 1, 2, 3, 4);
        var y = V(10, 7, 5, 2, 0);
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        double prev = interp.Interpolate(0.0);
        for (double t = 0.05; t <= 4.0; t += 0.05)
        {
            double curr = interp.Interpolate(t);
            Assert.True(curr <= prev + Tol,
                $"Monotonicity violated at t={t:F3}: {curr} > {prev}");
            prev = curr;
        }
    }

    [Fact]
    public void MonotoneCubic_FlatSegmentsPreserved()
    {
        // If two consecutive y-values are equal, the segment should be flat
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 1, 1, 1, 2);
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        // In the flat region [1,3], all values should be close to 1
        for (double t = 1.0; t <= 3.0; t += 0.1)
        {
            double val = interp.Interpolate(t);
            Assert.True(val >= 0.9 && val <= 1.1,
                $"Value {val} not near 1.0 in flat region at t={t}");
        }
    }

    [Fact]
    public void MonotoneCubic_ReproducesLinearFunction()
    {
        // f(x) = 4x + 1
        var x = V(0, 1, 2, 3, 4);
        var y = V(1, 5, 9, 13, 17);
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        Assert.Equal(4 * 0.5 + 1, interp.Interpolate(0.5), 1e-8);
        Assert.Equal(4 * 2.3 + 1, interp.Interpolate(2.3), 1e-8);
    }

    // ──────────────────────────────────────────────────────────
    // BARYCENTRIC RATIONAL INTERPOLATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Barycentric_ExactAtKnots()
    {
        var x = V(0, 1, 2, 3);
        var y = V(1, 4, 9, 16);
        var interp = new BarycentricRationalInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tol);
    }

    [Fact]
    public void Barycentric_EquivalentToLagrange()
    {
        // Barycentric with polynomial weights produces the same interpolant as Lagrange
        var x = V(0, 1, 2, 3);
        var y = V(2, 5, 10, 17);
        var bary = new BarycentricRationalInterpolation<double>(x, y);
        var lagr = new LagrangePolynomialInterpolation<double>(x, y);

        // At non-knot points they should agree (both compute the unique polynomial)
        for (double t = 0.1; t <= 2.9; t += 0.3)
            Assert.Equal(lagr.Interpolate(t), bary.Interpolate(t), 1e-8);
    }

    [Fact]
    public void Barycentric_ReproducesQuadratic()
    {
        // f(x) = x^2 with 3 points defines it uniquely
        var x = V(-1, 0, 1);
        var y = V(1, 0, 1);
        var interp = new BarycentricRationalInterpolation<double>(x, y);

        Assert.Equal(0.25, interp.Interpolate(0.5), 1e-8);
        Assert.Equal(0.25, interp.Interpolate(-0.5), 1e-8);
    }

    // ──────────────────────────────────────────────────────────
    // CROSS-METHOD COMPARISONS
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void CrossMethod_AllReproduceLinearData()
    {
        // All interpolation methods must exactly reproduce linear functions
        var x = V(0, 1, 2, 3, 4);
        var y = V(2, 5, 8, 11, 14); // f(x) = 3x + 2

        var linear = new LinearInterpolation<double>(x, y);
        var cubic = new CubicSplineInterpolation<double>(x, y);
        var lagrange = new LagrangePolynomialInterpolation<double>(x, y);
        var newton = new NewtonDividedDifferenceInterpolation<double>(x, y);
        var monotone = new MonotoneCubicInterpolation<double>(x, y);
        var bary = new BarycentricRationalInterpolation<double>(x, y);
        var nn = new NearestNeighborInterpolation<double>(x, y);
        var hermite = new HermiteInterpolation<double>(x, y, V(3, 3, 3, 3, 3));

        double[] testPoints = { 0.5, 1.5, 2.5, 3.5 };
        foreach (double t in testPoints)
        {
            double expected = 3 * t + 2;
            Assert.Equal(expected, linear.Interpolate(t), 1e-8);
            Assert.Equal(expected, cubic.Interpolate(t), 1e-8);
            Assert.Equal(expected, lagrange.Interpolate(t), 1e-8);
            Assert.Equal(expected, newton.Interpolate(t), 1e-8);
            Assert.Equal(expected, monotone.Interpolate(t), 1e-8);
            Assert.Equal(expected, bary.Interpolate(t), 1e-8);
            Assert.Equal(expected, hermite.Interpolate(t), 1e-8);
            // NN is step function, skip
        }
    }

    [Fact]
    public void CrossMethod_AllExactAtKnots()
    {
        // Every method must exactly reproduce data at knot points
        var x = V(0, 1, 2, 3, 4);
        var y = V(3, 7, 1, 9, 2);

        var linear = new LinearInterpolation<double>(x, y);
        var cubic = new CubicSplineInterpolation<double>(x, y);
        var lagrange = new LagrangePolynomialInterpolation<double>(x, y);
        var newton = new NewtonDividedDifferenceInterpolation<double>(x, y);
        var monotone = new MonotoneCubicInterpolation<double>(x, y);
        var bary = new BarycentricRationalInterpolation<double>(x, y);
        var nn = new NearestNeighborInterpolation<double>(x, y);
        var hermite = new HermiteInterpolation<double>(x, y, V(0, 0, 0, 0, 0));

        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], linear.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], cubic.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], lagrange.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], newton.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], monotone.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], bary.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], nn.Interpolate(x[i]), Tol);
            Assert.Equal(y[i], hermite.Interpolate(x[i]), Tol);
        }
    }

    [Fact]
    public void CrossMethod_HigherOrderMethodsBetterForSinusoid()
    {
        // Interpolating sin(x) at uniformly spaced points
        // Higher-order methods should be more accurate than linear at midpoints
        int n = 7;
        double[] xs = Enumerable.Range(0, n).Select(i => i * Math.PI / (n - 1)).ToArray();
        double[] ys = xs.Select(Math.Sin).ToArray();
        var x = V(xs);
        var y = V(ys);

        var linear = new LinearInterpolation<double>(x, y);
        var cubic = new CubicSplineInterpolation<double>(x, y);

        // At a midpoint between knots
        double testX = (xs[2] + xs[3]) / 2;
        double trueVal = Math.Sin(testX);

        double linearErr = Math.Abs(linear.Interpolate(testX) - trueVal);
        double cubicErr = Math.Abs(cubic.Interpolate(testX) - trueVal);

        // Cubic spline should be more accurate than linear for smooth functions
        Assert.True(cubicErr < linearErr,
            $"Cubic error {cubicErr} should be less than linear error {linearErr}");
    }

    // ──────────────────────────────────────────────────────────
    // EDGE CASES AND ERROR HANDLING
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Linear_DifferentLengthVectorsThrows()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearInterpolation<double>(V(1, 2, 3), V(1, 2)));
    }

    [Fact]
    public void CubicSpline_TwoPoints()
    {
        // Minimum valid case: 2 points should work as a line
        var x = V(0, 1);
        var y = V(0, 1);
        var interp = new CubicSplineInterpolation<double>(x, y);

        Assert.Equal(0.5, interp.Interpolate(0.5), 1e-8);
    }

    [Fact]
    public void Lagrange_TooFewPointsThrows()
    {
        Assert.Throws<ArgumentException>(() =>
            new LagrangePolynomialInterpolation<double>(V(1), V(1)));
    }

    [Fact]
    public void Newton_TooFewPointsThrows()
    {
        Assert.Throws<ArgumentException>(() =>
            new NewtonDividedDifferenceInterpolation<double>(V(1), V(1)));
    }

    [Fact]
    public void Barycentric_TooFewPointsThrows()
    {
        Assert.Throws<ArgumentException>(() =>
            new BarycentricRationalInterpolation<double>(V(1), V(1)));
    }

    [Fact]
    public void Hermite_MismatchedLengthsThrows()
    {
        Assert.Throws<ArgumentException>(() =>
            new HermiteInterpolation<double>(V(0, 1), V(0, 1), V(0)));
    }

    // ──────────────────────────────────────────────────────────
    // POLYNOMIAL PRECISION TESTS
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Lagrange_Degree4PolynomialWith5Points()
    {
        // f(x) = x^4 - 3x^2 + 2
        // 5 points should reproduce exactly
        double f(double x) => x * x * x * x - 3 * x * x + 2;
        var x = V(-2, -1, 0, 1, 2);
        var y = V(f(-2), f(-1), f(0), f(1), f(2));
        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        Assert.Equal(f(0.5), interp.Interpolate(0.5), Tol);
        Assert.Equal(f(-1.5), interp.Interpolate(-1.5), Tol);
        Assert.Equal(f(1.7), interp.Interpolate(1.7), Tol);
    }

    [Fact]
    public void Newton_Degree4PolynomialWith5Points()
    {
        // Same test as Lagrange - Newton should give identical results
        double f(double x) => x * x * x * x - 3 * x * x + 2;
        var x = V(-2, -1, 0, 1, 2);
        var y = V(f(-2), f(-1), f(0), f(1), f(2));
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);

        Assert.Equal(f(0.5), interp.Interpolate(0.5), Tol);
        Assert.Equal(f(-1.5), interp.Interpolate(-1.5), Tol);
        Assert.Equal(f(1.7), interp.Interpolate(1.7), Tol);
    }

    // ──────────────────────────────────────────────────────────
    // NUMERICAL DERIVATIVE VERIFICATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void CubicSpline_NumericalDerivativeContinuity()
    {
        // Verify that the numerical derivative is continuous across knots
        // (natural cubic spline has C2 continuity)
        var x = V(0, 1, 2, 3, 4);
        var y = V(0, 2, 1, 3, 0);
        var interp = new CubicSplineInterpolation<double>(x, y);

        double h = 1e-6;
        // Check at interior knots
        for (int i = 1; i < x.Length - 1; i++)
        {
            double xi = x[i];
            // Numerical derivative from left
            double derivLeft = (interp.Interpolate(xi) - interp.Interpolate(xi - h)) / h;
            // Numerical derivative from right
            double derivRight = (interp.Interpolate(xi + h) - interp.Interpolate(xi)) / h;
            // Should be approximately equal (C1 continuity)
            Assert.Equal(derivLeft, derivRight, 1e-3);
        }
    }

    [Fact]
    public void Hermite_NumericalDerivativeMatchesSlope()
    {
        // The numerical derivative at knots should match the specified slope
        var x = V(0, 1, 2);
        var y = V(0, 1, 0);
        var m = V(2.0, 0.0, -2.0);
        var interp = new HermiteInterpolation<double>(x, y, m);

        double h = 1e-7;
        // Check derivative at x=0 (should be ~2.0)
        double numDeriv0 = (interp.Interpolate(0.0 + h) - interp.Interpolate(0.0)) / h;
        Assert.Equal(2.0, numDeriv0, 1e-4);

        // Check derivative at x=1 (should be ~0.0)
        double numDeriv1 = (interp.Interpolate(1.0 + h) - interp.Interpolate(1.0 - h)) / (2 * h);
        Assert.Equal(0.0, numDeriv1, 1e-4);
    }

    // ──────────────────────────────────────────────────────────
    // STRESS TESTS WITH KNOWN FUNCTIONS
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void CubicSpline_SinApproximation()
    {
        // Cubic spline on sin(x) data with 11 points [0, pi]
        // Should approximate sin(x) well at intermediate points
        int n = 11;
        double[] xs = Enumerable.Range(0, n).Select(i => i * Math.PI / (n - 1)).ToArray();
        double[] ys = xs.Select(Math.Sin).ToArray();
        var interp = new CubicSplineInterpolation<double>(V(xs), V(ys));

        // Check at many intermediate points
        for (double t = 0.05; t < Math.PI - 0.05; t += 0.1)
        {
            double approx = interp.Interpolate(t);
            double exact = Math.Sin(t);
            Assert.True(Math.Abs(approx - exact) < 1e-4,
                $"sin({t:F2}): approx={approx:F6}, exact={exact:F6}, err={Math.Abs(approx - exact):E2}");
        }
    }

    [Fact]
    public void Lagrange_RungePhenomenon()
    {
        // Runge's function f(x) = 1/(1+25x^2) with equispaced points
        // Lagrange interpolation at equispaced points is known to diverge at boundaries
        // This test validates that the code correctly computes the (badly conditioned) polynomial
        double f(double x) => 1.0 / (1 + 25 * x * x);
        int n = 11;
        double[] xs = Enumerable.Range(0, n).Select(i => -1.0 + 2.0 * i / (n - 1)).ToArray();
        double[] ys = xs.Select(f).ToArray();
        var interp = new LagrangePolynomialInterpolation<double>(V(xs), V(ys));

        // At knots it should be exact
        for (int i = 0; i < n; i++)
            Assert.Equal(ys[i], interp.Interpolate(xs[i]), 1e-8);

        // Near boundaries, the error should be large (Runge phenomenon)
        double nearBoundary = 0.9;
        double lagrangeVal = interp.Interpolate(nearBoundary);
        double trueVal = f(nearBoundary);
        double error = Math.Abs(lagrangeVal - trueVal);
        // The error should be noticeable due to Runge phenomenon
        Assert.True(error > 1e-4, $"Expected Runge phenomenon error > 1e-4, got {error}");
    }
}
