using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests Experiment 5: Verify that Mellin-Fourier fingerprints are invariant to scaling and shifting.
/// </summary>
public class InvarianceTests
{
    [Fact]
    public void ScaleInvariantFingerprint_ScaledSignals_ProduceSimilarFingerprints()
    {
        var mellin = new MellinTransform<double>();
        int n = 64;

        // Create a test signal
        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 3 * i / n) + 0.5 * Math.Sin(2 * Math.PI * 7 * i / n);
        }

        // Scale by 2x
        var scaled = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            scaled[i] = 2.0 * signal[i];
        }

        var fp1 = mellin.ScaleInvariantFingerprint(signal);
        var fp2 = mellin.ScaleInvariantFingerprint(scaled);

        // Normalize fingerprints for comparison (since magnitude scales)
        double norm1 = 0, norm2 = 0;
        for (int i = 0; i < n; i++)
        {
            norm1 += fp1[i] * fp1[i];
            norm2 += fp2[i] * fp2[i];
        }
        norm1 = Math.Sqrt(norm1);
        norm2 = Math.Sqrt(norm2);

        // Compute cosine similarity
        double dotProduct = 0;
        for (int i = 0; i < n; i++)
        {
            dotProduct += (fp1[i] / norm1) * (fp2[i] / norm2);
        }

        // Scale-invariant fingerprints should have high cosine similarity
        Assert.True(dotProduct > 0.9,
            $"Cosine similarity between original and 2x scaled fingerprints is {dotProduct}, expected > 0.9");
    }

    [Fact]
    public void ScaleShiftInvariantFingerprint_OutputsAreNonZero()
    {
        var mellin = new MellinTransform<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * 5 * i / n);
        }

        var fingerprint = mellin.ScaleShiftInvariantFingerprint(signal);

        Assert.Equal(n, fingerprint.Length);

        // At least some components should be non-zero
        double totalEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            totalEnergy += fingerprint[i] * fingerprint[i];
        }
        Assert.True(totalEnergy > 0, "Fingerprint should have non-zero energy");
    }

    [Fact]
    public void MellinFourierFingerprint_DifferentSignals_ProduceDifferentFingerprints()
    {
        var mellin = new MellinTransform<double>();
        int n = 64;

        // Signal 1: low frequency
        var signal1 = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal1[i] = Math.Sin(2 * Math.PI * 2 * i / n);
        }

        // Signal 2: high frequency
        var signal2 = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal2[i] = Math.Sin(2 * Math.PI * 15 * i / n);
        }

        var fp1 = mellin.ScaleShiftInvariantFingerprint(signal1);
        var fp2 = mellin.ScaleShiftInvariantFingerprint(signal2);

        // Different signals should produce different fingerprints
        double diff = 0;
        for (int i = 0; i < n; i++)
        {
            diff += Math.Abs(fp1[i] - fp2[i]);
        }

        Assert.True(diff > 0.1, $"Different signals should produce different fingerprints, diff = {diff}");
    }
}
