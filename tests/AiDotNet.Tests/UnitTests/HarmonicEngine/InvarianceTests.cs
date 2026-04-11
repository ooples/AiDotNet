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

        // Scale-invariant fingerprints should have very high cosine similarity
        // (near 1.0 for pure amplitude scaling). 0.9 was too loose — the
        // transform is mathematically scale-invariant modulo numerical noise.
        Assert.True(dotProduct > 0.999,
            $"Cosine similarity between original and 2x scaled fingerprints is {dotProduct:F6}, expected > 0.999");
    }

    [Fact]
    public void ScaleShiftInvariantFingerprint_ScaledAndShifted_AreSimilar()
    {
        // The Mellin-Fourier (scale + shift) fingerprint should be invariant
        // to both amplitude scaling AND time shifts. Verify by generating a
        // base signal, a scaled version, and a shifted version — all three
        // should have similar fingerprints.
        var mellin = new MellinTransform<double>();
        int n = 64;

        var baseSignal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
            baseSignal[i] = Math.Cos(2 * Math.PI * 5 * i / n);

        // Scaled version: 3×
        var scaled = new Vector<double>(n);
        for (int i = 0; i < n; i++) scaled[i] = 3.0 * baseSignal[i];

        // Circularly shifted version
        int shift = 10;
        var shifted = new Vector<double>(n);
        for (int i = 0; i < n; i++) shifted[i] = baseSignal[(i + shift) % n];

        var fpBase = mellin.ScaleShiftInvariantFingerprint(baseSignal);
        var fpScaled = mellin.ScaleShiftInvariantFingerprint(scaled);
        var fpShifted = mellin.ScaleShiftInvariantFingerprint(shifted);

        Assert.Equal(n, fpBase.Length);

        // Each fingerprint should have non-zero energy
        double baseEnergy = 0, scaledEnergy = 0, shiftedEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            baseEnergy += fpBase[i] * fpBase[i];
            scaledEnergy += fpScaled[i] * fpScaled[i];
            shiftedEnergy += fpShifted[i] * fpShifted[i];
        }
        Assert.True(baseEnergy > 0, "Base fingerprint should have non-zero energy");
        Assert.True(scaledEnergy > 0, "Scaled fingerprint should have non-zero energy");
        Assert.True(shiftedEnergy > 0, "Shifted fingerprint should have non-zero energy");

        // Cosine similarity between base and each transformed version
        static double CosineSim(Vector<double> a, Vector<double> b, int n)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-15);
        }

        double simScaled = CosineSim(fpBase, fpScaled, n);
        double simShifted = CosineSim(fpBase, fpShifted, n);

        // Scaling should be handled perfectly by the Mellin part
        Assert.True(simScaled > 0.999,
            $"Scaled fingerprint cosine similarity {simScaled:F6} should be > 0.999");

        // Shifts are handled by the outer FFT magnitude. This is less tight
        // than pure scaling because of discrete-time boundary effects.
        Assert.True(simShifted > 0.9,
            $"Shifted fingerprint cosine similarity {simShifted:F6} should be > 0.9");
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
