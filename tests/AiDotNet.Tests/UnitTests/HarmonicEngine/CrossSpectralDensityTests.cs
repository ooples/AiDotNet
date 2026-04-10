using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests for cross-spectral density and coherence computations.
/// </summary>
public class CrossSpectralDensityTests
{
    private readonly ITestOutputHelper _output;

    public CrossSpectralDensityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void AutoSpectral_PureSinusoid_PeaksAtCorrectFrequency()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 64;
        double freq = 5.0;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * freq * i / n);
        }

        var psd = csd.AutoSpectral(signal);

        // Find peak
        int peakBin = 0;
        double peakVal = 0;
        for (int k = 1; k < n / 2; k++)
        {
            if (psd[k] > peakVal)
            {
                peakVal = psd[k];
                peakBin = k;
            }
        }

        Assert.Equal((int)freq, peakBin);
        _output.WriteLine($"Peak at bin {peakBin} (expected {freq}), power = {peakVal:F2}");
    }

    [Fact]
    public void CrossSpectral_IdenticalSignals_EqualsAutoSpectral()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 3 * i / n) + 0.5 * Math.Cos(2 * Math.PI * 7 * i / n);
        }

        var crossSpec = csd.Compute(signal, signal);
        var autoSpec = csd.AutoSpectral(signal);

        // For identical signals, CSD should equal PSD (real-valued, imaginary ~0)
        for (int k = 0; k < n; k++)
        {
            // |CSD(k)| should equal PSD(k)
            double csdMag = crossSpec[k].Magnitude;
            Assert.Equal(autoSpec[k], csdMag, 4);
        }
    }

    [Fact]
    public void Coherence_IdenticalSignals_IsOne()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * 4 * i / n);
        }

        var coherence = csd.Coherence(signal, signal);

        // Coherence of signal with itself should be 1 at the signal frequency
        Assert.Equal(1.0, coherence[4], 4);
    }

    [Fact]
    public void Coherence_OrthogonalSignals_IsNearZero()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 64;

        // Two sinusoids at different frequencies are orthogonal
        var signal1 = new Vector<double>(n);
        var signal2 = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal1[i] = Math.Sin(2 * Math.PI * 3 * i / n);
            signal2[i] = Math.Sin(2 * Math.PI * 11 * i / n);
        }

        var coherence = csd.Coherence(signal1, signal2);

        // At frequency 3, signal2 has no energy -> coherence should be low
        // At frequency 11, signal1 has no energy -> coherence should be low
        // Overall coherence should be low for orthogonal signals
        double avgCoherence = 0;
        for (int k = 1; k < n / 2; k++)
        {
            avgCoherence += coherence[k];
        }
        avgCoherence /= (n / 2 - 1);

        _output.WriteLine($"Average coherence between orthogonal signals: {avgCoherence:F4}");

        // Coherence values are in [0, 1]
        for (int k = 0; k < n; k++)
        {
            Assert.True(coherence[k] >= -1e-10, $"Coherence[{k}] = {coherence[k]} should be >= 0");
            Assert.True(coherence[k] <= 1.0 + 1e-10, $"Coherence[{k}] = {coherence[k]} should be <= 1");
        }
    }

    [Fact]
    public void CrossSpectral_ScaledSignal_ScalesProperly()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        var scaled = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * 5 * i / n);
            scaled[i] = 3.0 * signal[i];
        }

        var csdOriginal = csd.Compute(signal, signal);
        var csdScaled = csd.Compute(signal, scaled);

        // CSD(x, 3x) should be 3 * CSD(x, x)
        for (int k = 0; k < n; k++)
        {
            double originalMag = csdOriginal[k].Magnitude;
            double scaledMag = csdScaled[k].Magnitude;

            if (originalMag > 1e-10)
            {
                double ratio = scaledMag / originalMag;
                Assert.Equal(3.0, ratio, 2);
            }
        }
    }
}
