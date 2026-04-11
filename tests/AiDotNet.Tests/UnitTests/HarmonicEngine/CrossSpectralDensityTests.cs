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
        int n = 4096; // Long enough for Welch's method (multiple segments)
        int segLen = 256;

        // Broadband signal: sum of many frequencies so most bins have energy
        var rng = new Random(42);
        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = rng.NextDouble() * 2 - 1; // white noise
        }

        var coherence = csd.Coherence(signal, signal, segLen);

        // Self-coherence on a broadband signal should be ~1 at every bin
        // (Welch's method with windowing still gives numerical γ²=1 because
        // |Sxy|² = Sxx·Syy when y = x, regardless of windowing)
        for (int k = 1; k < segLen / 2; k++)
        {
            double c = Convert.ToDouble(coherence[k]);
            Assert.True(c > 0.9,
                $"Self-coherence at bin {k} should be ~1, got {c:F4}");
        }
    }

    [Fact]
    public void Coherence_OrthogonalSignals_IsNearZero()
    {
        var csd = new CrossSpectralDensity<double>();
        int n = 1024; // Long enough for Welch's method

        // Two sinusoids plus independent noise — genuinely uncorrelated
        var rng = new Random(42);
        var signal1 = new Vector<double>(n);
        var signal2 = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal1[i] = Math.Sin(2 * Math.PI * 12 * i / n) + 0.5 * (rng.NextDouble() * 2 - 1);
            signal2[i] = Math.Sin(2 * Math.PI * 47 * i / n) + 0.5 * (rng.NextDouble() * 2 - 1);
        }

        var coherence = csd.Coherence(signal1, signal2);
        int segLen = coherence.Length;

        double avgCoherence = 0;
        for (int k = 1; k < segLen / 2; k++)
        {
            avgCoherence += coherence[k];
        }
        avgCoherence /= (segLen / 2 - 1);

        _output.WriteLine($"Average coherence between uncorrelated signals: {avgCoherence:F4}");

        // Average coherence between uncorrelated signals should be low
        Assert.True(avgCoherence < 0.4,
            $"Average coherence between uncorrelated signals should be low, got {avgCoherence:F4}");

        // Coherence values are in [0, 1]
        for (int k = 0; k < segLen; k++)
        {
            Assert.True(coherence[k] >= -1e-10, $"Coherence[{k}] = {coherence[k]} should be >= 0");
            Assert.True(coherence[k] <= 1.0 + 1e-10, $"Coherence[{k}] = {coherence[k]} should be <= 1");
        }
    }

    [Fact]
    public void Coherence_LinearlyRelatedSignals_IsHigh()
    {
        // Welch's method should detect linear relationships:
        // y = 2*x + noise should have coherence near 1 at the signal frequencies
        var csd = new CrossSpectralDensity<double>();
        int n = 1024;

        var rng = new Random(7);
        var x = new Vector<double>(n);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double s = Math.Sin(2 * Math.PI * 25 * i / n);
            x[i] = s + 0.1 * (rng.NextDouble() * 2 - 1);
            y[i] = 2.0 * s + 0.1 * (rng.NextDouble() * 2 - 1);
        }

        var coherence = csd.Coherence(x, y);

        // At the signal frequency, coherence should be high (near 1)
        // Find the bin corresponding to f=25 in the segment
        int segLen = coherence.Length;
        double maxCoherence = 0;
        for (int k = 1; k < segLen / 2; k++)
        {
            if (coherence[k] > maxCoherence) maxCoherence = coherence[k];
        }

        _output.WriteLine($"Max coherence for y=2x+noise: {maxCoherence:F4}");
        Assert.True(maxCoherence > 0.8,
            $"Linearly related signals should have high coherence at signal frequency, got max={maxCoherence:F4}");
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
