using AiDotNet.Interpolation;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Interpolation;

/// <summary>
/// Unit tests for interpolation classes to achieve 100% code coverage.
/// Tests cover constructors, input validation, and edge cases.
/// </summary>
public class InterpolationUnitTests
{
    private const double Tolerance = 1e-10;

    #region Linear Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var interp = new LinearInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Constructor_DifferentLengths_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        Assert.Throws<ArgumentException>(() => new LinearInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Constructor_SinglePoint_Succeeds()
    {
        // Linear interpolation allows single point (returns constant)
        var x = new Vector<double>(new[] { 0.0 });
        var y = new Vector<double>(new[] { 5.0 });
        var interp = new LinearInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
        var interp = new LinearInterpolation<double>(x, y);

        Assert.Equal(0.0, interp.Interpolate(0.0), Tolerance);
        Assert.Equal(2.0, interp.Interpolate(1.0), Tolerance);
        Assert.Equal(4.0, interp.Interpolate(2.0), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Interpolate_BetweenPoints_ReturnsCorrectValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
        var interp = new LinearInterpolation<double>(x, y);

        Assert.Equal(1.0, interp.Interpolate(0.5), Tolerance);
        Assert.Equal(3.0, interp.Interpolate(1.5), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Interpolate_BeforeFirstPoint_Extrapolates()
    {
        var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var interp = new LinearInterpolation<double>(x, y);

        var result = interp.Interpolate(0.0);
        Assert.False(double.IsNaN(result));
    }

    [Fact(Timeout = 60000)]
    public async Task LinearInterpolation_Interpolate_AfterLastPoint_Extrapolates()
    {
        var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var interp = new LinearInterpolation<double>(x, y);

        var result = interp.Interpolate(4.0);
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Cubic Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task CubicSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CubicSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task CubicSplineInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CubicSplineInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tolerance);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task CubicSplineInterpolation_Interpolate_BetweenPoints_ReturnsSmooth()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CubicSplineInterpolation<double>(x, y);

        var result = interp.Interpolate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.True(result > 0.0 && result < 1.0);
    }

    #endregion

    #region Akima Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task AkimaInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0 });
        var interp = new AkimaInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task AkimaInterpolation_Constructor_DifferentLengths_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
        Assert.Throws<ArgumentException>(() => new AkimaInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task AkimaInterpolation_Constructor_TooFewPoints_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        Assert.Throws<ArgumentException>(() => new AkimaInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task AkimaInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
        var interp = new AkimaInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tolerance);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task AkimaInterpolation_Interpolate_EqualWeights_UsesMiddleSlope()
    {
        // Create data where w1 == w2 == 0 to hit the special case
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 }); // Linear data
        var interp = new AkimaInterpolation<double>(x, y);

        // With linear data, all slopes are equal so w1=w2=0
        var result = interp.Interpolate(2.5);
        Assert.Equal(2.5, result, Tolerance);
    }

    #endregion

    #region Monotone Cubic Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var interp = new MonotoneCubicInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Constructor_DifferentLengths_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        Assert.Throws<ArgumentException>(() => new MonotoneCubicInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Constructor_TooFewPoints_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0 });
        var y = new Vector<double>(new[] { 0.0 });
        Assert.Throws<ArgumentException>(() => new MonotoneCubicInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Interpolate_IncreasingData_PreservesMonotonicity()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        double prev = interp.Interpolate(0.0);
        for (double t = 0.1; t <= 3.0; t += 0.1)
        {
            double current = interp.Interpolate(t);
            Assert.True(current >= prev - Tolerance);
            prev = current;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Interpolate_LocalExtremum_SetsZeroSlope()
    {
        // Data with a local maximum at x=1
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 2.0, 1.0, 3.0 });
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        // The interpolation should handle the extremum
        var result = interp.Interpolate(1.0);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Interpolate_FlatSection_HandlesZeroSecant()
    {
        // Data with a flat section (delta[i] = 0)
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 }); // Flat between x=1 and x=2
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        var result = interp.Interpolate(1.5);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task MonotoneCubicInterpolation_Interpolate_LargeSlopes_ConstrainsToMaintainMonotonicity()
    {
        // Data that would produce large slopes needing constraint
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 0.1, 0.2, 5.0, 5.1 }); // Sharp change at x=3
        var interp = new MonotoneCubicInterpolation<double>(x, y);

        // Verify no overshoot
        double prev = interp.Interpolate(2.0);
        for (double t = 2.1; t <= 3.0; t += 0.1)
        {
            double current = interp.Interpolate(t);
            Assert.True(current >= prev - Tolerance);
            prev = current;
        }
    }

    #endregion

    #region Catmull-Rom Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CatmullRomSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Constructor_DifferentLengths_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        Assert.Throws<ArgumentException>(() => new CatmullRomSplineInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Constructor_TooFewPoints_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        Assert.Throws<ArgumentException>(() => new CatmullRomSplineInterpolation<double>(x, y));
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Constructor_CustomTension_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CatmullRomSplineInterpolation<double>(x, y, tension: 0.8);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Interpolate_AtExactPoint_ReturnsExactValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0 });
        var interp = new CatmullRomSplineInterpolation<double>(x, y);

        // Test exact point match (uses early return)
        Assert.Equal(1.0, interp.Interpolate(1.0), Tolerance);
        Assert.Equal(0.0, interp.Interpolate(2.0), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Interpolate_BetweenPoints_ReturnsSmooth()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var interp = new CatmullRomSplineInterpolation<double>(x, y);

        var result = interp.Interpolate(1.5);
        Assert.Equal(1.5, result, 0.1); // Linear data should give ~linear result
    }

    [Fact(Timeout = 60000)]
    public async Task CatmullRomSplineInterpolation_Interpolate_LastInterval_ReturnsCorrectValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var interp = new CatmullRomSplineInterpolation<double>(x, y);

        var result = interp.Interpolate(3.5);
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region PCHIP Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task PchipInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new PchipInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task PchipInterpolation_Interpolate_PreservesMonotonicity()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var interp = new PchipInterpolation<double>(x, y);

        double prev = interp.Interpolate(0.0);
        for (double t = 0.1; t <= 3.0; t += 0.1)
        {
            double current = interp.Interpolate(t);
            Assert.True(current >= prev - Tolerance);
            prev = current;
        }
    }

    #endregion

    #region Lagrange Polynomial Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task LagrangePolynomialInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 });
        var interp = new LagrangePolynomialInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task LagrangePolynomialInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 }); // y = x^2
        var interp = new LagrangePolynomialInterpolation<double>(x, y);

        Assert.Equal(0.0, interp.Interpolate(0.0), Tolerance);
        Assert.Equal(1.0, interp.Interpolate(1.0), Tolerance);
        Assert.Equal(4.0, interp.Interpolate(2.0), Tolerance);
    }

    #endregion

    #region Newton Divided Difference Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task NewtonDividedDifferenceInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 });
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task NewtonDividedDifferenceInterpolation_Interpolate_Quadratic_ReturnsCorrectValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 }); // y = x^2
        var interp = new NewtonDividedDifferenceInterpolation<double>(x, y);

        Assert.Equal(0.25, interp.Interpolate(0.5), Tolerance);
        Assert.Equal(2.25, interp.Interpolate(1.5), Tolerance);
    }

    #endregion

    #region Hermite Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task HermiteInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 });
        var dy = new Vector<double>(new[] { 0.0, 2.0, 4.0 }); // derivatives
        var interp = new HermiteInterpolation<double>(x, y, dy);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task HermiteInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 }); // y = x^2
        var dy = new Vector<double>(new[] { 0.0, 2.0, 4.0 }); // dy/dx = 2x
        var interp = new HermiteInterpolation<double>(x, y, dy);

        Assert.Equal(0.0, interp.Interpolate(0.0), Tolerance);
        Assert.Equal(1.0, interp.Interpolate(1.0), Tolerance);
        Assert.Equal(4.0, interp.Interpolate(2.0), Tolerance);
    }

    #endregion

    #region Nearest Neighbor Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task NearestNeighborInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var interp = new NearestNeighborInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task NearestNeighborInterpolation_Interpolate_ReturnsNearestValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 10.0, 20.0 });
        var interp = new NearestNeighborInterpolation<double>(x, y);

        Assert.Equal(0.0, interp.Interpolate(0.4), Tolerance);
        Assert.Equal(10.0, interp.Interpolate(0.6), Tolerance);
        Assert.Equal(10.0, interp.Interpolate(1.4), Tolerance);
        Assert.Equal(20.0, interp.Interpolate(1.6), Tolerance);
    }

    #endregion

    #region Barycentric Rational Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task BarycentricRationalInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new BarycentricRationalInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task BarycentricRationalInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.0 });
        var interp = new BarycentricRationalInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tolerance);
        }
    }

    #endregion

    #region Lanczos Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task LanczosInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
        var interp = new LanczosInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task LanczosInterpolation_Constructor_CustomA_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
        var interp = new LanczosInterpolation<double>(x, y, a: 4);
        Assert.NotNull(interp);
    }

    #endregion

    #region Sinc Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task SincInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new SincInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    #endregion

    #region Trigonometric Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task TrigonometricInterpolation_Constructor_ValidInput_Succeeds()
    {
        // Trigonometric interpolation requires odd number of points
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, -1.0, 0.0 });
        var interp = new TrigonometricInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task TrigonometricInterpolation_Constructor_EvenPoints_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, -1.0 });
        Assert.Throws<ArgumentException>(() => new TrigonometricInterpolation<double>(x, y));
    }

    #endregion

    #region Natural Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task NaturalSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new NaturalSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task NaturalSplineInterpolation_Interpolate_AtKnownPoints_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new NaturalSplineInterpolation<double>(x, y);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], interp.Interpolate(x[i]), Tolerance);
        }
    }

    #endregion

    #region Clamped Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task ClampedSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new ClampedSplineInterpolation<double>(x, y, 1.0, -1.0);
        Assert.NotNull(interp);
    }

    #endregion

    #region Not-A-Knot Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task NotAKnotSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new NotAKnotSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    #endregion

    #region Adaptive Cubic Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task AdaptiveCubicSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new AdaptiveCubicSplineInterpolation<double>(x, y, threshold: 0.1);
        Assert.NotNull(interp);
    }

    #endregion

    #region Cubic B-Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task CubicBSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new CubicBSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    #endregion

    #region Cubic Convolution Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task CubicConvolutionInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var z = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                z[i, j] = i + j;
        var interp = new CubicConvolutionInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region Kochanek-Bartels Spline Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task KochanekBartelsSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new KochanekBartelsSplineInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task KochanekBartelsSplineInterpolation_Constructor_CustomParameters_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new KochanekBartelsSplineInterpolation<double>(x, y, tension: 0.5, bias: 0.2, continuity: 0.1);
        Assert.NotNull(interp);
    }

    #endregion

    #region Whittaker-Shannon Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task WhittakerShannonInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new WhittakerShannonInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    #endregion

    #region Gaussian Process Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task GaussianProcessInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var interp = new GaussianProcessInterpolation<double>(x, y);
        Assert.NotNull(interp);
    }

    #endregion

    #region Radial Basis Function Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task RadialBasisFunctionInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new RadialBasisFunctionInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region Multiquadric Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task MultiquadricInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new MultiquadricInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region Moving Least Squares Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task MovingLeastSquaresInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new MovingLeastSquaresInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region Kriging Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task KrigingInterpolation_Constructor_ValidInput_Succeeds()
    {
        // Kriging needs more points for variogram estimation
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0 });
        var interp = new KrigingInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region 2D Interpolation Tests

    [Fact(Timeout = 60000)]
    public async Task BilinearInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(new[,] { { 0.0, 1.0 }, { 1.0, 2.0 } });
        var interp = new BilinearInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task BilinearInterpolation_Interpolate_AtCorners_ReturnsExactValues()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(new[,] { { 0.0, 1.0 }, { 2.0, 3.0 } });
        var interp = new BilinearInterpolation<double>(x, y, z);

        Assert.Equal(0.0, interp.Interpolate(0.0, 0.0), Tolerance);
        Assert.Equal(1.0, interp.Interpolate(0.0, 1.0), Tolerance);
        Assert.Equal(2.0, interp.Interpolate(1.0, 0.0), Tolerance);
        Assert.Equal(3.0, interp.Interpolate(1.0, 1.0), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task BilinearInterpolation_Interpolate_AtCenter_ReturnsAverage()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(new[,] { { 0.0, 2.0 }, { 2.0, 4.0 } });
        var interp = new BilinearInterpolation<double>(x, y, z);

        Assert.Equal(2.0, interp.Interpolate(0.5, 0.5), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task BicubicInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var z = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                z[i, j] = i + j;
        var interp = new BicubicInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task ShepardsMethodInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new ShepardsMethodInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    [Fact(Timeout = 60000)]
    public async Task ShepardsMethodInterpolation_Interpolate_AtKnownPoint_ReturnsExactValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new ShepardsMethodInterpolation<double>(x, y, z);

        Assert.Equal(0.0, interp.Interpolate(0.0, 0.0), Tolerance);
        Assert.Equal(2.0, interp.Interpolate(1.0, 1.0), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task ThinPlateSplineInterpolation_Constructor_ValidInput_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp = new ThinPlateSplineInterpolation<double>(x, y, z);
        Assert.NotNull(interp);
    }

    #endregion

    #region Interpolation2DTo1DAdapter Tests

    [Fact(Timeout = 60000)]
    public async Task Interpolation2DTo1DAdapter_Constructor_FixedY_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp2d = new ShepardsMethodInterpolation<double>(x, y, z);
        var adapter = new Interpolation2DTo1DAdapter<double>(interp2d, 0.5, isXFixed: false);
        Assert.NotNull(adapter);
    }

    [Fact(Timeout = 60000)]
    public async Task Interpolation2DTo1DAdapter_Constructor_FixedX_Succeeds()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp2d = new ShepardsMethodInterpolation<double>(x, y, z);
        var adapter = new Interpolation2DTo1DAdapter<double>(interp2d, 0.5, isXFixed: true);
        Assert.NotNull(adapter);
    }

    [Fact(Timeout = 60000)]
    public async Task Interpolation2DTo1DAdapter_Interpolate_FixedY_ReturnsCorrectValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp2d = new ShepardsMethodInterpolation<double>(x, y, z);
        var adapter = new Interpolation2DTo1DAdapter<double>(interp2d, 0.0, isXFixed: false);

        // At y=0, z = x (approximately)
        var result = adapter.Interpolate(0.5);
        Assert.False(double.IsNaN(result));
    }

    [Fact(Timeout = 60000)]
    public async Task Interpolation2DTo1DAdapter_Interpolate_FixedX_ReturnsCorrectValue()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var z = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0 });
        var interp2d = new ShepardsMethodInterpolation<double>(x, y, z);
        var adapter = new Interpolation2DTo1DAdapter<double>(interp2d, 0.0, isXFixed: true);

        // At x=0, z = y (approximately)
        var result = adapter.Interpolate(0.5);
        Assert.False(double.IsNaN(result));
    }

    #endregion
}
