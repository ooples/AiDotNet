using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Deep math-correctness integration tests for FastFourierTransform.
/// Verifies DFT identities, Parseval's theorem, linearity, shift, and hand-calculated values.
/// </summary>
public class FFTDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    private static readonly FastFourierTransform<double> FFT = new();

    #region Forward / Inverse Roundtrip

    [Fact]
    public void FFT_Forward_Inverse_Roundtrip_PowerOf2()
    {
        // x -> FFT -> IFFT -> x
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var freq = FFT.Forward(x);
        var recovered = FFT.Inverse(freq);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(x[i], recovered[i], Tolerance);
    }

    [Fact]
    public void FFT_Forward_Inverse_Roundtrip_Size4()
    {
        var x = new Vector<double>(new double[] { 3, -1, 4, -2 });
        var freq = FFT.Forward(x);
        var recovered = FFT.Inverse(freq);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(x[i], recovered[i], Tolerance);
    }

    [Fact]
    public void FFT_Forward_Inverse_Roundtrip_Size2()
    {
        var x = new Vector<double>(new double[] { 5, 3 });
        var freq = FFT.Forward(x);
        var recovered = FFT.Inverse(freq);

        Assert.Equal(5.0, recovered[0], Tolerance);
        Assert.Equal(3.0, recovered[1], Tolerance);
    }

    [Fact]
    public void FFT_Forward_Inverse_Roundtrip_Size1()
    {
        var x = new Vector<double>(new double[] { 7 });
        var freq = FFT.Forward(x);
        var recovered = FFT.Inverse(freq);

        Assert.Equal(7.0, recovered[0], Tolerance);
    }

    #endregion

    #region DC Component (X[0] = sum of all samples)

    [Fact]
    public void FFT_DCComponent_EqualsSum()
    {
        // X[0] = sum(x[n]) for DFT (no normalization in forward)
        var x = new Vector<double>(new double[] { 1, 2, 3, 4 });
        var freq = FFT.Forward(x);

        double sum = 1 + 2 + 3 + 4;
        Assert.Equal(sum, freq[0].Real, Tolerance);
        Assert.Equal(0.0, freq[0].Imaginary, Tolerance);
    }

    [Fact]
    public void FFT_DCComponent_EqualsSum_Size8()
    {
        var x = new Vector<double>(new double[] { 2, -1, 3, 0, 5, -2, 1, 4 });
        var freq = FFT.Forward(x);

        double sum = 2 + (-1) + 3 + 0 + 5 + (-2) + 1 + 4;
        Assert.Equal(sum, freq[0].Real, Tolerance);
        Assert.Equal(0.0, freq[0].Imaginary, Tolerance);
    }

    #endregion

    #region Constant Signal

    [Fact]
    public void FFT_ConstantSignal_OnlyDCComponent()
    {
        // Constant signal c: X[0] = N*c, X[k] = 0 for k > 0
        double c = 5.0;
        int n = 8;
        var x = new Vector<double>(n);
        for (int i = 0; i < n; i++) x[i] = c;

        var freq = FFT.Forward(x);

        Assert.Equal(n * c, freq[0].Real, Tolerance);
        Assert.Equal(0.0, freq[0].Imaginary, Tolerance);

        for (int k = 1; k < n; k++)
        {
            Assert.Equal(0.0, freq[k].Real, Tolerance);
            Assert.Equal(0.0, freq[k].Imaginary, Tolerance);
        }
    }

    #endregion

    #region Parseval's Theorem

    [Fact]
    public void FFT_ParsevalsTheorem_EnergyConservation()
    {
        // sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
        var x = new Vector<double>(new double[] { 1, -2, 3, -4, 5, -6, 7, -8 });
        var freq = FFT.Forward(x);

        double timeDomainEnergy = 0;
        for (int i = 0; i < x.Length; i++)
            timeDomainEnergy += x[i] * x[i];

        double freqDomainEnergy = 0;
        for (int k = 0; k < freq.Length; k++)
        {
            double re = freq[k].Real;
            double im = freq[k].Imaginary;
            freqDomainEnergy += re * re + im * im;
        }
        freqDomainEnergy /= x.Length;

        Assert.Equal(timeDomainEnergy, freqDomainEnergy, LooseTolerance);
    }

    #endregion

    #region Linearity

    [Fact]
    public void FFT_Linearity_FFTofAx_Plus_By_Equals_aFFTx_Plus_bFFTy()
    {
        // FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
        double a = 2.0, b = -3.0;
        var x = new Vector<double>(new double[] { 1, 2, 3, 4 });
        var y = new Vector<double>(new double[] { 4, 3, 2, 1 });

        // Compute a*x + b*y
        var combined = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
            combined[i] = a * x[i] + b * y[i];

        var freqCombined = FFT.Forward(combined);
        var freqX = FFT.Forward(x);
        var freqY = FFT.Forward(y);

        for (int k = 0; k < 4; k++)
        {
            double expectedReal = a * freqX[k].Real + b * freqY[k].Real;
            double expectedImag = a * freqX[k].Imaginary + b * freqY[k].Imaginary;
            Assert.Equal(expectedReal, freqCombined[k].Real, Tolerance);
            Assert.Equal(expectedImag, freqCombined[k].Imaginary, Tolerance);
        }
    }

    #endregion

    #region Hand-Calculated DFT

    [Fact]
    public void FFT_HandCalculated_Size2()
    {
        // DFT of [a, b]:
        // X[0] = a + b, X[1] = a - b
        var x = new Vector<double>(new double[] { 3, 7 });
        var freq = FFT.Forward(x);

        Assert.Equal(10.0, freq[0].Real, Tolerance); // 3 + 7
        Assert.Equal(0.0, freq[0].Imaginary, Tolerance);
        Assert.Equal(-4.0, freq[1].Real, Tolerance); // 3 - 7
        Assert.Equal(0.0, freq[1].Imaginary, Tolerance);
    }

    [Fact]
    public void FFT_HandCalculated_Size4()
    {
        // DFT of [1, 0, 0, 0] (impulse):
        // X[k] = 1 for all k (since sum of e^0 = 1 for each freq)
        var x = new Vector<double>(new double[] { 1, 0, 0, 0 });
        var freq = FFT.Forward(x);

        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(1.0, freq[k].Real, Tolerance);
            Assert.Equal(0.0, freq[k].Imaginary, Tolerance);
        }
    }

    [Fact]
    public void FFT_HandCalculated_Size4_AllOnes()
    {
        // DFT of [1, 1, 1, 1]:
        // X[0] = 4, X[1] = X[2] = X[3] = 0
        var x = new Vector<double>(new double[] { 1, 1, 1, 1 });
        var freq = FFT.Forward(x);

        Assert.Equal(4.0, freq[0].Real, Tolerance);
        Assert.Equal(0.0, freq[0].Imaginary, Tolerance);
        for (int k = 1; k < 4; k++)
        {
            Assert.Equal(0.0, freq[k].Real, Tolerance);
            Assert.Equal(0.0, freq[k].Imaginary, Tolerance);
        }
    }

    [Fact]
    public void FFT_HandCalculated_Size4_Alternating()
    {
        // DFT of [1, -1, 1, -1]:
        // X[0] = 0, X[1] = 0, X[2] = 4, X[3] = 0
        // (Nyquist frequency component)
        var x = new Vector<double>(new double[] { 1, -1, 1, -1 });
        var freq = FFT.Forward(x);

        Assert.Equal(0.0, freq[0].Real, Tolerance);
        Assert.Equal(0.0, freq[1].Real, Tolerance);
        Assert.Equal(4.0, freq[2].Real, Tolerance); // Nyquist
        Assert.Equal(0.0, freq[3].Real, Tolerance);
    }

    #endregion

    #region Real Signal Symmetry

    [Fact]
    public void FFT_RealSignal_ConjugateSymmetry()
    {
        // For real input: X[k] = conj(X[N-k])
        var x = new Vector<double>(new double[] { 2, -1, 3, 0, 5, -2, 1, 4 });
        var freq = FFT.Forward(x);

        int n = x.Length;
        for (int k = 1; k < n / 2; k++)
        {
            Assert.Equal(freq[k].Real, freq[n - k].Real, Tolerance);
            Assert.Equal(freq[k].Imaginary, -freq[n - k].Imaginary, Tolerance);
        }
    }

    #endregion

    #region Pure Sine Wave

    [Fact]
    public void FFT_PureSineWave_SingleFrequencyPeak()
    {
        // x[n] = sin(2*pi*f*n/N), f=1 -> peak at k=1 and k=N-1
        int n = 8;
        var x = new Vector<double>(n);
        for (int i = 0; i < n; i++)
            x[i] = Math.Sin(2 * Math.PI * i / n);

        var freq = FFT.Forward(x);

        // DC and other bins should be ~0
        Assert.Equal(0.0, freq[0].Real, LooseTolerance);

        // Bin 1: should have imaginary component = -N/2 = -4
        double mag1 = Math.Sqrt(freq[1].Real * freq[1].Real + freq[1].Imaginary * freq[1].Imaginary);
        Assert.Equal(4.0, mag1, LooseTolerance); // N/2

        // Bin N-1: conjugate symmetric
        double magN1 = Math.Sqrt(freq[n - 1].Real * freq[n - 1].Real + freq[n - 1].Imaginary * freq[n - 1].Imaginary);
        Assert.Equal(4.0, magN1, LooseTolerance); // N/2

        // Other bins should be ~0
        for (int k = 2; k < n - 1; k++)
        {
            double mag = Math.Sqrt(freq[k].Real * freq[k].Real + freq[k].Imaginary * freq[k].Imaginary);
            Assert.True(mag < LooseTolerance, $"|X[{k}]| = {mag} should be ~0");
        }
    }

    #endregion

    #region Pure Cosine Wave

    [Fact]
    public void FFT_PureCosineWave_RealSymmetricPeaks()
    {
        // x[n] = cos(2*pi*f*n/N), f=1 -> real peaks at k=1 and k=N-1
        int n = 8;
        var x = new Vector<double>(n);
        for (int i = 0; i < n; i++)
            x[i] = Math.Cos(2 * Math.PI * i / n);

        var freq = FFT.Forward(x);

        // Bin 1 and N-1 should be N/2 = 4, real part only
        Assert.Equal(4.0, freq[1].Real, LooseTolerance); // N/2
        Assert.Equal(0.0, freq[1].Imaginary, LooseTolerance);
        Assert.Equal(4.0, freq[n - 1].Real, LooseTolerance); // N/2
        Assert.Equal(0.0, freq[n - 1].Imaginary, LooseTolerance);
    }

    #endregion

    #region Zero Signal

    [Fact]
    public void FFT_ZeroSignal_AllZeroSpectrum()
    {
        var x = new Vector<double>(new double[] { 0, 0, 0, 0, 0, 0, 0, 0 });
        var freq = FFT.Forward(x);

        for (int k = 0; k < freq.Length; k++)
        {
            Assert.Equal(0.0, freq[k].Real, Tolerance);
            Assert.Equal(0.0, freq[k].Imaginary, Tolerance);
        }
    }

    #endregion

    #region Scaling Property

    [Fact]
    public void FFT_ScalingProperty_FFTofScaledSignal()
    {
        // FFT(c*x) = c * FFT(x)
        double c = 3.5;
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var scaled = new Vector<double>(8);
        for (int i = 0; i < 8; i++) scaled[i] = c * x[i];

        var freqX = FFT.Forward(x);
        var freqScaled = FFT.Forward(scaled);

        for (int k = 0; k < 8; k++)
        {
            Assert.Equal(c * freqX[k].Real, freqScaled[k].Real, Tolerance);
            Assert.Equal(c * freqX[k].Imaginary, freqScaled[k].Imaginary, Tolerance);
        }
    }

    #endregion

    #region Time Reversal

    [Fact]
    public void FFT_TimeReversal_ConjugateSpectrum()
    {
        // If Y[n] = X[N-n], then FFT(Y)[k] = conj(FFT(X)[k])
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        int n = x.Length;

        // Create time-reversed signal
        var reversed = new Vector<double>(n);
        reversed[0] = x[0]; // x[0] stays at index 0
        for (int i = 1; i < n; i++)
            reversed[i] = x[n - i];

        var freqX = FFT.Forward(x);
        var freqReversed = FFT.Forward(reversed);

        for (int k = 0; k < n; k++)
        {
            Assert.Equal(freqX[k].Real, freqReversed[k].Real, Tolerance);
            Assert.Equal(-freqX[k].Imaginary, freqReversed[k].Imaginary, Tolerance);
        }
    }

    #endregion
}
