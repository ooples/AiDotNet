using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests the analytic signal (Hilbert transform) computation.
/// </summary>
public class AnalyticSignalTests
{
    [Fact]
    public void Compute_CosineInput_ProducesCorrectAnalyticSignal()
    {
        // For cos(wt), the analytic signal should be cos(wt) + i*sin(wt) = e^(iwt)
        var analytic = new AnalyticSignal<double>();
        int n = 64;
        double freq = 4.0; // 4 cycles in the window

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * freq * i / n);
        }

        var result = analytic.Compute(signal);

        // The real part should match the original cosine
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(signal[i], result[i].Real, 3);
        }

        // The imaginary part should approximate sin(wt)
        for (int i = 1; i < n - 1; i++) // Skip edges due to edge effects
        {
            double expectedImag = Math.Sin(2 * Math.PI * freq * i / n);
            Assert.Equal(expectedImag, MathHelper.GetNumericOperations<double>().ToDouble(result[i].Imaginary), 1);
        }
    }

    [Fact]
    public void InstantaneousAmplitude_ConstantAmplitude_ReturnsUniform()
    {
        var analytic = new AnalyticSignal<double>();
        int n = 64;
        double amp = 3.0;
        double freq = 5.0;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = amp * Math.Cos(2 * Math.PI * freq * i / n);
        }

        var amplitude = analytic.InstantaneousAmplitude(signal);

        // Should be approximately constant at 'amp' (away from edges)
        for (int i = 5; i < n - 5; i++)
        {
            Assert.Equal(amp, amplitude[i], 0); // Within 1.0 tolerance
        }
    }

    [Fact]
    public void InstantaneousFrequency_SingleFrequency_ReturnsConstant()
    {
        var analytic = new AnalyticSignal<double>();
        int n = 128;
        double freq = 8.0; // 8 cycles in window

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * freq * i / n);
        }

        var instFreq = analytic.InstantaneousFrequency(signal);

        // Should be approximately constant (normalized frequency)
        double expectedNormFreq = freq / n;
        int midStart = n / 4;
        int midEnd = 3 * n / 4;

        for (int i = midStart; i < midEnd && i < instFreq.Length; i++)
        {
            Assert.Equal(expectedNormFreq, instFreq[i], 2);
        }
    }
}
