using System;
using AiDotNet.Kernels;
using Xunit;

namespace AiDotNetTests.UnitTests.Kernels;

/// <summary>
/// Reference values (SciPy / closed forms) for kernel formulas corrected in the element-wise sweep.
/// </summary>
public class KernelReferenceValueTests
{
    private const double Tolerance = 1e-9;

    [Fact]
    public void BesselKernel_OrderZero_MatchesScipyJ0()
    {
        // scipy.special.jv(0, 1.0) = 0.7651976865579666; kernel = J_0(r / sigma) / (r / sigma)^0
        var kernel = new BesselKernel<double>(order: 0.0, sigma: 1.0);
        var x1 = new Vector<double>(new[] { 0.0 });
        var x2 = new Vector<double>(new[] { 1.0 });

        Assert.Equal(0.7651976865579666, kernel.Calculate(x1, x2), 1e-10);
    }

    [Fact]
    public void BesselKernel_OrderOne_MatchesScipyJ1()
    {
        // scipy.special.jv(1, 2.0) / 2.0 = 0.2883624038784368
        var kernel = new BesselKernel<double>(order: 1.0, sigma: 1.0);
        var x1 = new Vector<double>(new[] { 0.0, 0.0 });
        var x2 = new Vector<double>(new[] { 2.0, 0.0 });

        Assert.Equal(0.2883624038784368, kernel.Calculate(x1, x2), 1e-10);
    }

    [Fact]
    public void GibbsKernel_TwoDimensions_UsesDimensionPowerOfNormalisation()
    {
        // Rasmussen & Williams eq. 4.32 with l(x) = 1 + x_0 in D = 2:
        // l1 = 1, l2 = 2, r^2 = 1 -> (2*1*2/5)^(2/2) * exp(-1/5) = 0.8 * exp(-0.2)
        var kernel = new GibbsKernel<double>(x => 1.0 + x[0]);
        var x1 = new Vector<double>(new[] { 0.0, 0.0 });
        var x2 = new Vector<double>(new[] { 1.0, 0.0 });

        Assert.Equal(0.8 * Math.Exp(-0.2), kernel.Calculate(x1, x2), Tolerance);
    }

    [Fact]
    public void SplineKernel_MatchesVapnikSplineFormula()
    {
        // Per dimension with m = min(x, y): 1 + xy + xy*m - (x + y)/2 * m^2 + m^3/3
        // (1, 2): 1 + 2 + 2 - 1.5 + 1/3; (0.5, 1): 1 + 0.5 + 0.25 - 0.1875 + 0.125/3
        var kernel = new SplineKernel<double>();
        var x1 = new Vector<double>(new[] { 1.0, 0.5 });
        var x2 = new Vector<double>(new[] { 2.0, 1.0 });

        double expected = (1 + 2 + 2 - 1.5 + 1.0 / 3.0) * (1 + 0.5 + 0.25 - 0.1875 + 0.125 / 3.0);
        Assert.Equal(expected, kernel.Calculate(x1, x2), Tolerance);
    }

    [Fact]
    public void SpectralDeltaKernel_SamePointInSeveralDimensions_ReturnsVariance()
    {
        // k(x, x) = sigma^2 for the spectral mixture product over dimensions (Wilson & Adams 2013).
        var kernel = new SpectralDeltaKernel<double>(frequency: 1.0, bandwidth: 0.1, variance: 2.5);
        var x = new Vector<double>(new[] { 0.3, -1.2, 4.0 });

        Assert.Equal(2.5, kernel.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void BSpline_Cubic_IsCenteredAndSymmetric()
    {
        // Centered cubic B-spline: B3(0) = 2/3 and B3(+-0.5) = 23/48.
        var kernel = new BSplineKernel<double>(3, 1.0);
        var origin = new Vector<double>(new[] { 0.0, 0.0 });
        var shifted = new Vector<double>(new[] { 0.5, 0.0 });

        Assert.Equal(4.0 / 9.0, kernel.Calculate(origin, origin), 12);
        Assert.Equal(23.0 / 48.0 * 2.0 / 3.0, kernel.Calculate(origin, shifted), 12);
        Assert.Equal(kernel.Calculate(origin, shifted), kernel.Calculate(shifted, origin), 15);
    }
}
