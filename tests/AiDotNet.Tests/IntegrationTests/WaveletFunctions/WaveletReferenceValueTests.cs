using AiDotNet.WaveletFunctions;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.WaveletFunctions;

/// <summary>
/// Reference values for wavelet definitions, derived by hand from the published formulas.
/// </summary>
public class WaveletReferenceValueTests
{
    private const double Tolerance = 1e-9;

    // ================================================================
    // B-SPLINE: cardinal B-spline N_n via Cox-de Boor, binomial filter taps
    // ================================================================

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order1_IsHatFunction()
    {
        // N_1(x) = x on [0, 1), 2 - x on [1, 2)
        var wavelet = new BSplineWavelet<double>(order: 1);
        Assert.Equal(0.5, wavelet.Calculate(0.5), Tolerance);
        Assert.Equal(1.0, wavelet.Calculate(1.0), Tolerance);
        Assert.Equal(0.25, wavelet.Calculate(1.75), Tolerance);
        Assert.Equal(0.0, wavelet.Calculate(2.5), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order2_QuadraticValues()
    {
        // N_2(x) = x^2 / 2 on [0, 1), (-2x^2 + 6x - 3) / 2 on [1, 2): N_2(0.5) = 1/8, N_2(1.5) = 3/4
        var wavelet = new BSplineWavelet<double>(order: 2);
        Assert.Equal(0.125, wavelet.Calculate(0.5), Tolerance);
        Assert.Equal(0.75, wavelet.Calculate(1.5), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order3_CubicValues()
    {
        // Cubic cardinal B-spline on [0, 4]: N_3(1) = 1/6, N_3(2) = 2/3
        var wavelet = new BSplineWavelet<double>(order: 3);
        Assert.Equal(1.0 / 6.0, wavelet.Calculate(1.0), Tolerance);
        Assert.Equal(2.0 / 3.0, wavelet.Calculate(2.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order2_FiltersAreNormalizedBinomials()
    {
        // Low pass: C(2, k) = [1, 2, 1] / sqrt(6); high pass: C(3, k) with alternating signs / sqrt(20)
        var wavelet = new BSplineWavelet<double>(order: 2);
        var low = wavelet.GetScalingCoefficients();
        var high = wavelet.GetWaveletCoefficients();

        double lowNorm = Math.Sqrt(6.0);
        Assert.Equal(3, low.Length);
        Assert.Equal(1.0 / lowNorm, low[0], Tolerance);
        Assert.Equal(2.0 / lowNorm, low[1], Tolerance);
        Assert.Equal(1.0 / lowNorm, low[2], Tolerance);

        double highNorm = Math.Sqrt(20.0);
        Assert.Equal(4, high.Length);
        Assert.Equal(1.0 / highNorm, high[0], Tolerance);
        Assert.Equal(-3.0 / highNorm, high[1], Tolerance);
        Assert.Equal(3.0 / highNorm, high[2], Tolerance);
        Assert.Equal(-1.0 / highNorm, high[3], Tolerance);
    }

    // ================================================================
    // PAUL: psi(t) = 2^m i^m m! / sqrt(pi (2m)!) * (1 - i t)^-(m + 1)
    // ================================================================

    [Fact(Timeout = 120000)]
    public async Task Paul_Order4_MatchesClosedForm()
    {
        // m = 4: factor = 16 * 24 / sqrt(pi * 40320) = 1.0789368501515768 and i^4 = 1.
        // (1 - i)^-5 = 2^-2.5 * e^(i 5 pi / 4) = -0.125 - 0.125i, so Re psi(1) = -0.1348671062689471.
        // (1 - 2i)^-5 = (41 + 38i) / 3125, giving Re psi(2) = 1.0789368501515768 * 41 / 3125.
        var wavelet = new PaulWavelet<double>(order: 4);
        Assert.Equal(1.0789368501515768, wavelet.Calculate(0.0), 1e-9);
        Assert.Equal(-0.1348671062689471, wavelet.Calculate(1.0), 1e-9);
        Assert.Equal(0.014155651473988687, wavelet.Calculate(2.0), 1e-9);
    }
}
