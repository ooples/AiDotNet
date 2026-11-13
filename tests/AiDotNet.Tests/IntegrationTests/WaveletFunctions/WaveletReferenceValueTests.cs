using AiDotNet.Tensors.LinearAlgebra;
using System;
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
        await Task.Yield();
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
        await Task.Yield();
        // N_2(x) = x^2 / 2 on [0, 1), (-2x^2 + 6x - 3) / 2 on [1, 2): N_2(0.5) = 1/8, N_2(1.5) = 3/4
        var wavelet = new BSplineWavelet<double>(order: 2);
        Assert.Equal(0.125, wavelet.Calculate(0.5), Tolerance);
        Assert.Equal(0.75, wavelet.Calculate(1.5), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order3_CubicValues()
    {
        await Task.Yield();
        // Cubic cardinal B-spline on [0, 4]: N_3(1) = 1/6, N_3(2) = 2/3
        var wavelet = new BSplineWavelet<double>(order: 3);
        Assert.Equal(1.0 / 6.0, wavelet.Calculate(1.0), Tolerance);
        Assert.Equal(2.0 / 3.0, wavelet.Calculate(2.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BSpline_Order2_FiltersAreNormalizedBinomials()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
        // He_5(1) = 1 - 10 + 15 = 6, so psi(1) = -6 e^-0.5 / (sqrt(120) * 8)
        var wavelet = new DOGWavelet<double>(order: 5);
        double expected = -6.0 * Math.Exp(-0.5) / (Math.Sqrt(120.0) * 8.0);
        Assert.Equal(expected, wavelet.Calculate(1.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Dog_Order6_UsesHermitePolynomial()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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

    [Fact(Timeout = 120000)]
    public async Task BattleLemarie_ReconstructIsAdjointOfDecompose()
    {
        await Task.Yield();
        // Reconstructing a unit approximation impulse places the scaling filter at the start of the signal:
        // x[m] = sum_j h[j] up(a)[m - j] = h[m]. The previous index arithmetic never matched, returning zeros.
        var wavelet = new BattleLemarieWavelet<double>(order: 1);
        var h = wavelet.GetScalingCoefficients();
        var impulse = new Vector<double>(8);
        impulse[0] = 1.0;

        var reconstructed = wavelet.Reconstruct(impulse, new Vector<double>(8));
        Assert.Equal(16, reconstructed.Length);
        for (int m = 0; m < 16; m++)
        {
            double expected = m < h.Length ? h[m] : 0.0;
            Assert.Equal(expected, reconstructed[m], 1e-12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FejerKorovkin_ReconstructIsAdjointOfDecompose()
    {
        await Task.Yield();
        // A unit approximation impulse reconstructs to the scaling filter, a unit detail impulse to the wavelet
        // filter (the previous index arithmetic never matched and returned zeros).
        var wavelet = new FejérKorovkinWavelet<double>(order: 4);
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();
        var impulse = new Vector<double>(4);
        impulse[0] = 1.0;

        var fromApproximation = wavelet.Reconstruct(impulse, new Vector<double>(4));
        var fromDetail = wavelet.Reconstruct(new Vector<double>(4), impulse);
        for (int m = 0; m < 8; m++)
        {
            Assert.Equal(m < h.Length ? h[m] : 0.0, fromApproximation[m], 1e-12);
            Assert.Equal(m < g.Length ? g[m] : 0.0, fromDetail[m], 1e-12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ComplexMorlet_ScalingFilter_IsSinc()
    {
        await Task.Yield();
        // 64 taps at x = (i - 32) / 16 normalized by the sum of magnitudes; ratios to the centre tap are sinc values.
        var wavelet = new ComplexMorletWavelet<double>();
        var h = wavelet.GetScalingCoefficients();
        Assert.Equal(64, h.Length);
        Assert.Equal(2.0 / Math.PI, h[40].Real / h[32].Real, Tolerance);
        Assert.Equal(-2.0 / (3.0 * Math.PI), h[56].Real / h[32].Real, Tolerance);
        Assert.Equal(0.0, h[40].Imaginary, Tolerance);
    }

    // ================================================================
    // EXTENSIBILITY: filters and transforms are built through the overridable scalar Calculate
    // ================================================================

    private sealed class FlatGaussianWavelet : GaussianWavelet<double>
    {
        public override double Calculate(double x) => 2.0;
    }

    private sealed class FlatMexicanHatWavelet : MexicanHatWavelet<double>
    {
        public override double Calculate(double x) => 3.0;
    }

    [Fact(Timeout = 120000)]
    public async Task GaussianSubclass_OverridingCalculate_ShapesFiltersAndTransforms()
    {
        await Task.Yield();
        // The derivative filter is -(x / sigma^2) * Calculate(x), so it follows the override too.
        var wavelet = new FlatGaussianWavelet();
        var scaling = wavelet.GetScalingCoefficients();
        var detailFilter = wavelet.GetWaveletCoefficients();
        Assert.Equal(101, scaling.Length);
        for (int i = 0; i < scaling.Length; i++)
        {
            Assert.Equal(2.0, scaling[i], Tolerance);
            Assert.Equal(-(i - 50) * 2.0, detailFilter[i], Tolerance);
        }

        var (approximation, _) = wavelet.Decompose(new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        Assert.Equal(2.0, approximation[0], Tolerance);
        Assert.Equal(6.0, approximation[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task MexicanHatSubclass_OverridingCalculate_ShapesScalingFilter()
    {
        await Task.Yield();
        var wavelet = new FlatMexicanHatWavelet();
        var scaling = wavelet.GetScalingCoefficients();
        for (int i = 0; i < scaling.Length; i++)
        {
            Assert.Equal(3.0, scaling[i], Tolerance);
        }

        var (approximation, _) = wavelet.Decompose(new Vector<double>(new[] { 1.0, 2.0 }));
        Assert.Equal(3.0, approximation[0], Tolerance);
        Assert.Equal(6.0, approximation[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task GaussianWavelet_TensorCalculate_MatchesScalarCalculate()
    {
        await Task.Yield();
        var wavelet = new GaussianWavelet<double>(sigma: 1.5);
        var x = new Tensor<double>(new[] { -2.0, -0.5, 0.0, 1.25, 3.0 }, new[] { 5 });
        var values = wavelet.Calculate(x);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(wavelet.Calculate(x[i]), values[i], 1e-14);
        }
    }

    [Theory(Timeout = 120000)]
    [InlineData(4, 12)]
    [InlineData(6, 12)]
    [InlineData(8, 7)]
    [InlineData(14, 10)]
    [InlineData(22, 9)]
    public async Task FejerKorovkin_PublishedOrders_ReconstructInvertsDecompose(int order, int digits)
    {
        await Task.Yield();
        // The published taps are orthonormal to their printed precision (fk8 to ~1e-9), which bounds the error.
        var wavelet = new FejérKorovkinWavelet<double>(order);
        var values = new double[48];
        for (int i = 0; i < values.Length; i++) values[i] = Math.Sin(0.37 * i) + 0.25 * (i % 5) - (i == 7 ? 2.0 : 0.0);
        var signal = new Vector<double>(values);

        var (approximation, detail) = wavelet.Decompose(signal);
        var restored = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(signal.Length, restored.Length);
        for (int i = 0; i < signal.Length; i++)
        {
            Assert.Equal(signal[i], restored[i], digits);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FejerKorovkin_UnpublishedOrder_ReconstructDoesNotThrow()
    {
        await Task.Yield();
        var wavelet = new FejérKorovkinWavelet<double>(10);
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });

        var (approximation, detail) = wavelet.Decompose(signal);
        var restored = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(signal.Length, restored.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Daubechies_CascadeConvolvesWithTheScalingFilter()
    {
        await Task.Yield();
        // Reference: master's scalar cascade, new[(2i + k) mod 2n] += h[k] * values[i], seven iterations.
        var wavelet = new DaubechiesWavelet<double>(4);
        var h = wavelet.GetScalingCoefficients();
        double Cascade(double t)
        {
            var values = new double[1 << 7];
            values[0] = 1;
            for (int iter = 0; iter < 7; iter++)
            {
                var next = new double[values.Length * 2];
                for (int i = 0; i < values.Length; i++)
                {
                    for (int k = 0; k < h.Length; k++)
                    {
                        next[(2 * i + k) % next.Length] += h[k] * values[i];
                    }
                }

                values = next;
            }

            return values[(int)(t * values.Length)];
        }

        foreach (var x in new[] { 0.3, 1.45, 2.7 })
        {
            int k = (int)Math.Floor(x);
            Assert.Equal(h[k] * Cascade(x - k), wavelet.Calculate(x), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SincCoefficients_Decimal_DoNotDivideByZeroAtTheCentre()
    {
        await Task.Yield();
        Assert.Equal(64, new ContinuousMexicanHatWavelet<decimal>().GetScalingCoefficients().Length);
        var shannon = new ShannonWavelet<decimal>();
        Assert.True(shannon.GetScalingCoefficients().Length > 0);
        Assert.True(shannon.GetWaveletCoefficients().Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FejerKorovkin_FloatMatchesDouble()
    {
        await Task.Yield();
        // The coefficient sum cancels to zero; float residue (~1.5e-8) used as a divisor blew the values up.
        double expected = new FejérKorovkinWavelet<double>(4).Calculate(0.5);
        float actual = new FejérKorovkinWavelet<float>(4).Calculate(0.5f);

        Assert.Equal(expected, actual, 5);
    }

    [Fact(Timeout = 120000)]
    public async Task PointwiseReconstruct_DecimalZeroDenominator_FallsBackWithoutThrowing()
    {
        await Task.Yield();
        var zeros = new Tensor<decimal>(new[] { 2 });
        var restored = WaveletEngineOps<decimal>.PointwiseReconstruct(
            new Vector<decimal>(new[] { 1m, 2m }), new Vector<decimal>(new[] { 3m, 4m }), zeros, zeros);

        Assert.Equal(4m, restored[0]);
        Assert.Equal(6m, restored[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task SpectralReconstruct_MismatchedLengths_Throw()
    {
        await Task.Yield();
        var four = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
        var two = new Vector<double>(new[] { 0.0, 0.0 });

        Assert.Throws<ArgumentException>(() => new MeyerWavelet<double>().Reconstruct(four, two));
        Assert.Throws<ArgumentException>(() => new MorletWavelet<double>().Reconstruct(four, two));
        Assert.Throws<ArgumentException>(() => new PaulWavelet<double>().Reconstruct(four, two));
    }
}
