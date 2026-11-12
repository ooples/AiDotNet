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
        // (1 - 2i)^-5 = (41 - 38i) / 3125, giving Re psi(2) = 1.0789368501515768 * 41 / 3125.
        var wavelet = new PaulWavelet<double>(order: 4);
        Assert.Equal(1.0789368501515768, wavelet.Calculate(0.0), 1e-9);
        Assert.Equal(-0.1348671062689471, wavelet.Calculate(1.0), 1e-9);
        Assert.Equal(0.014155651473988687, wavelet.Calculate(2.0), 1e-9);
    }

    // ================================================================
    // DOG: d^n/dx^n exp(-x^2/2) = (-1)^n He_n(x) exp(-x^2/2), scaled by 1 / (sqrt(n!) 2^((n+1)/2))
    // ================================================================

    [Fact(Timeout = 120000)]
    public async Task Dog_Order5_UsesHermitePolynomial()
    {
        // He_5(1) = 1 - 10 + 15 = 6, so psi(1) = -6 e^-0.5 / (sqrt(120) * 8)
        var wavelet = new DOGWavelet<double>(order: 5);
        double expected = -6.0 * Math.Exp(-0.5) / (Math.Sqrt(120.0) * 8.0);
        Assert.Equal(expected, wavelet.Calculate(1.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Dog_Order6_UsesHermitePolynomial()
    {
        // He_6(0.5) = 0.5^6 - 15 * 0.5^4 + 45 * 0.5^2 - 15 = -4.671875
        var wavelet = new DOGWavelet<double>(order: 6);
        double expected = -4.671875 * Math.Exp(-0.125) / (Math.Sqrt(720.0) * Math.Pow(2.0, 3.5));
        Assert.Equal(expected, wavelet.Calculate(0.5), Tolerance);
    }

    // ================================================================
    // CONTINUOUS MEXICAN HAT: the scaling filter is the documented sinc(x) = sin(pi x) / (pi x)
    // ================================================================

    [Fact(Timeout = 120000)]
    public async Task ContinuousMexicanHat_ScalingFilter_IsSinc()
    {
        // 64 taps at x = (i - 32) / 16, normalized by the sum of magnitudes, so ratios to the centre tap are
        // sinc values: x = 0.5 -> 2 / pi, x = 1 -> 0, x = 1.5 -> -2 / (3 pi).
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var h = wavelet.GetScalingCoefficients();
        Assert.Equal(64, h.Length);
        Assert.Equal(2.0 / Math.PI, h[40] / h[32], Tolerance);
        Assert.Equal(0.0, h[48] / h[32], Tolerance);
        Assert.Equal(-2.0 / (3.0 * Math.PI), h[56] / h[32], Tolerance);
    }

    // ================================================================
    // ORTHOGONAL FILTER BANKS: periodic analysis followed by its adjoint reconstructs the signal exactly
    // ================================================================

    private static readonly double[] RoundTripSignal = { 1, 5, 2, 8, 3, 7, 4, 6, 9, 2, 5, 1, 7, 3, 8, 4 };

    private static void AssertRoundTrip(Vector<double> reconstructed, double tolerance)
    {
        Assert.Equal(RoundTripSignal.Length, reconstructed.Length);
        for (int i = 0; i < RoundTripSignal.Length; i++)
        {
            Assert.Equal(RoundTripSignal[i], reconstructed[i], tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Daubechies_ReconstructInvertsDecompose()
    {
        var wavelet = new DaubechiesWavelet<double>();
        var (approximation, detail) = wavelet.Decompose(new Vector<double>(RoundTripSignal));
        AssertRoundTrip(wavelet.Reconstruct(approximation, detail), 1e-12);
    }

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public async Task Symlet_ReconstructInvertsDecompose(int order)
    {
        // The filters are the PyWavelets symN taps, which are orthonormal to double precision.
        var wavelet = new SymletWavelet<double>(order);
        var (approximation, detail) = wavelet.Decompose(new Vector<double>(RoundTripSignal));
        AssertRoundTrip(wavelet.Reconstruct(approximation, detail), 1e-10);
    }

    [Theory(Timeout = 120000)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public async Task Coiflet_ScalingFilterIsOrthonormal(int order)
    {
        // An orthonormal scaling filter sums to sqrt(2) and is orthogonal to its even shifts.
        var h = new CoifletWavelet<double>(order).GetScalingCoefficients();
        Assert.Equal(6 * order, h.Length);

        double sum = 0;
        for (int i = 0; i < h.Length; i++)
        {
            sum += h[i];
        }

        Assert.Equal(Math.Sqrt(2.0), sum, 1e-8);
        for (int shift = 2; shift < h.Length; shift += 2)
        {
            double dot = 0;
            for (int i = 0; i + shift < h.Length; i++)
            {
                dot += h[i] * h[i + shift];
            }

            Assert.Equal(0.0, dot, 1e-8);
        }
    }
}
