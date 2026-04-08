using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the FastFourierTransform<T> class covering forward/inverse transforms,
/// signal reconstruction, and mathematical properties.
/// </summary>
public class FastFourierTransformIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;

    #region Forward Transform

    [Fact]
    public void FFT_Forward_ConstantSignal_HasOnlyDCComponent()
    {
        // A constant signal should have all energy in the DC (first) component
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(new double[] { 5, 5, 5, 5 });

        var spectrum = fft.Forward(signal);

        // DC component should be 5 * 4 = 20
        Assert.Equal(20.0, spectrum[0].Real, Tolerance);
        Assert.Equal(0.0, spectrum[0].Imaginary, Tolerance);

        // All other components should be zero
        for (int i = 1; i < spectrum.Length; i++)
        {
            Assert.Equal(0.0, spectrum[i].Magnitude, Tolerance);
        }
    }

    [Fact]
    public void FFT_Forward_AlternatingSignal_HasOnlyNyquistComponent()
    {
        // Signal alternating between +1 and -1 has all energy at Nyquist frequency
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(new double[] { 1, -1, 1, -1 });

        var spectrum = fft.Forward(signal);

        // DC component should be zero
        Assert.Equal(0.0, spectrum[0].Magnitude, Tolerance);

        // Nyquist component (index n/2) should have all energy
        int nyquist = signal.Length / 2;
        Assert.True(spectrum[nyquist].Magnitude > 0);
    }

    [Fact]
    public void FFT_Forward_SinglePulse_HasFlatMagnitudeSpectrum()
    {
        // A single pulse (impulse) should have equal magnitude across all frequencies
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(new double[] { 1, 0, 0, 0, 0, 0, 0, 0 });

        var spectrum = fft.Forward(signal);

        // All components should have magnitude 1
        for (int i = 0; i < spectrum.Length; i++)
        {
            Assert.Equal(1.0, spectrum[i].Magnitude, Tolerance);
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void FFT_Forward_ZeroSignal_ProducesZeroSpectrum(int length)
    {
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(length);
        for (int i = 0; i < length; i++) signal[i] = 0;

        var spectrum = fft.Forward(signal);

        for (int i = 0; i < spectrum.Length; i++)
        {
            Assert.Equal(0.0, spectrum[i].Magnitude, Tolerance);
        }
    }

    #endregion

    #region Inverse Transform

    [Fact]
    public void FFT_Inverse_DCOnlySpectrum_ProducesConstantSignal()
    {
        var fft = new FastFourierTransform<double>();
        var spectrum = new Vector<Complex<double>>(4);
        spectrum[0] = new Complex<double>(8, 0); // DC = 8, so signal will be 8/4 = 2
        for (int i = 1; i < spectrum.Length; i++)
        {
            spectrum[i] = new Complex<double>(0, 0);
        }

        var signal = fft.Inverse(spectrum);

        for (int i = 0; i < signal.Length; i++)
        {
            Assert.Equal(2.0, signal[i], Tolerance);
        }
    }

    #endregion

    #region Round-Trip (Forward then Inverse)

    [Fact]
    public void FFT_RoundTrip_PreservesSignal()
    {
        var fft = new FastFourierTransform<double>();
        var original = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        var spectrum = fft.Forward(original);
        var reconstructed = fft.Inverse(spectrum);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], LooseTolerance);
        }
    }

    [Fact]
    public void FFT_RoundTrip_SinusoidalSignal_PreservesSignal()
    {
        var fft = new FastFourierTransform<double>();
        int n = 16;
        var original = new Vector<double>(n);

        // Create a simple sinusoidal signal
        for (int i = 0; i < n; i++)
        {
            original[i] = Math.Sin(2 * Math.PI * i / n);
        }

        var spectrum = fft.Forward(original);
        var reconstructed = fft.Inverse(spectrum);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], LooseTolerance);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void FFT_RoundTrip_RandomSignal_PreservesSignal(int length)
    {
        var fft = new FastFourierTransform<double>();
        var random = new Random(42);
        var original = new Vector<double>(length);

        for (int i = 0; i < length; i++)
        {
            original[i] = random.NextDouble() * 10 - 5;
        }

        var spectrum = fft.Forward(original);
        var reconstructed = fft.Inverse(spectrum);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], LooseTolerance);
        }
    }

    #endregion

    #region Parseval's Theorem

    [Fact]
    public void FFT_ParsevalsTheorem_EnergyPreserved()
    {
        // Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        // Compute time-domain energy
        double timeDomainEnergy = 0;
        for (int i = 0; i < signal.Length; i++)
        {
            timeDomainEnergy += signal[i] * signal[i];
        }

        // Compute frequency-domain energy
        var spectrum = fft.Forward(signal);
        double freqDomainEnergy = 0;
        for (int i = 0; i < spectrum.Length; i++)
        {
            double mag = spectrum[i].Magnitude;
            freqDomainEnergy += mag * mag;
        }
        freqDomainEnergy /= signal.Length;

        Assert.Equal(timeDomainEnergy, freqDomainEnergy, LooseTolerance);
    }

    #endregion

    #region Linearity

    [Fact]
    public void FFT_Linearity_SumOfSignalsEqualsSumOfTransforms()
    {
        var fft = new FastFourierTransform<double>();
        var signal1 = new Vector<double>(new double[] { 1, 2, 3, 4 });
        var signal2 = new Vector<double>(new double[] { 4, 3, 2, 1 });

        // Transform each signal individually
        var spectrum1 = fft.Forward(signal1);
        var spectrum2 = fft.Forward(signal2);

        // Sum signals then transform
        var sumSignal = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            sumSignal[i] = signal1[i] + signal2[i];
        }
        var sumSpectrum = fft.Forward(sumSignal);

        // FFT(signal1 + signal2) should equal FFT(signal1) + FFT(signal2)
        for (int i = 0; i < sumSpectrum.Length; i++)
        {
            var expected = spectrum1[i] + spectrum2[i];
            Assert.Equal(expected.Real, sumSpectrum[i].Real, LooseTolerance);
            Assert.Equal(expected.Imaginary, sumSpectrum[i].Imaginary, LooseTolerance);
        }
    }

    [Fact]
    public void FFT_Linearity_ScaledSignalEqualsScaledTransform()
    {
        var fft = new FastFourierTransform<double>();
        double scale = 3.0;
        var signal = new Vector<double>(new double[] { 1, 2, 3, 4 });

        // Transform original signal
        var spectrum = fft.Forward(signal);

        // Scale signal then transform
        var scaledSignal = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            scaledSignal[i] = signal[i] * scale;
        }
        var scaledSpectrum = fft.Forward(scaledSignal);

        // FFT(scale * signal) should equal scale * FFT(signal)
        for (int i = 0; i < scaledSpectrum.Length; i++)
        {
            Assert.Equal(spectrum[i].Real * scale, scaledSpectrum[i].Real, LooseTolerance);
            Assert.Equal(spectrum[i].Imaginary * scale, scaledSpectrum[i].Imaginary, LooseTolerance);
        }
    }

    #endregion

    #region Conjugate Symmetry for Real Signals

    [Fact]
    public void FFT_RealSignal_HasConjugateSymmetry()
    {
        // For real signals: X[k] = conj(X[N-k])
        var fft = new FastFourierTransform<double>();
        var signal = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        int n = signal.Length;

        var spectrum = fft.Forward(signal);

        for (int k = 1; k < n / 2; k++)
        {
            var xk = spectrum[k];
            var xnk = spectrum[n - k];

            // X[k] should equal conjugate of X[N-k]
            Assert.Equal(xk.Real, xnk.Real, LooseTolerance);
            Assert.Equal(xk.Imaginary, -xnk.Imaginary, LooseTolerance);
        }
    }

    #endregion
}
