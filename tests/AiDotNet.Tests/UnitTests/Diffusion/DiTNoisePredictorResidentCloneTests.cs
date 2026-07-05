using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Regression tests for issue #1764: cloning a foundation-scale DiT after an fp16-resident eval forward
/// used to diverge (~8% on DIAMOND) because the resident fp16 weight master was not part of the
/// parameter round-trip — the probe forward downcast the clone's random init to the resident master, and
/// the subsequent CopyParametersFrom wrote the (transient) fp32 scratch while the stale random master was
/// what the next forward actually consumed. These tests force the resident path at small scale via the
/// internal <see cref="DiTNoisePredictor{T}.ResidentThresholdOverrideForTests"/> hook so the fix is guarded
/// without allocating a real &gt;1B-parameter model (which OOMs the CI runner and only runs nightly).
/// </summary>
public class DiTNoisePredictorResidentCloneTests
{
    private static DiTNoisePredictor<float> CreateResidentPredictor(int seed)
    {
        var p = new DiTNoisePredictor<float>(
            inputChannels: 16, hiddenSize: 32, numLayers: 2, numHeads: 2,
            patchSize: 2, contextDim: 64, latentSpatialSize: 8, seed: seed);
        // Any nonzero param count exceeds this, so the model takes the fp16-resident eval path.
        p.ResidentThresholdOverrideForTests = 1_000L;
        return p;
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var d = t.GetCpuData();
        var rng = new Random(seed);
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void Clone_UnderFp16Resident_ReproducesSourceOutput()
    {
        var source = CreateResidentPredictor(seed: 123);
        var input = Rand(new[] { 1, 16, 8, 8 }, seed: 7);
        var conditioning = Rand(new[] { 1, 1, 64 }, seed: 9); // exercise the resident cross-attention path too

        var o1 = source.PredictNoise(input, timestep: 0, conditioning: conditioning);
        var clone = (DiTNoisePredictor<float>)source.Clone();
        var o2 = clone.PredictNoise(input, timestep: 0, conditioning: conditioning);

        Assert.Equal(o1.Length, o2.Length);

        // The source output must be a real (finite, non-degenerate) signal, else "equal" would be trivial.
        double maxAbs = 0;
        for (int i = 0; i < o1.Length; i++)
        {
            // net471 has no double.IsFinite; !IsNaN && !IsInfinity is the equivalent (both exist since .NET 2.0).
            double v = (double)o1[i];
            Assert.True(!double.IsNaN(v) && !double.IsInfinity(v), $"source output[{i}] not finite");
            maxAbs = Math.Max(maxAbs, Math.Abs(v));
        }
        Assert.True(maxAbs > 1e-6, "source output is all ~zero — test would be vacuous");

        // Source and clone both run fp16-resident with identical masters → bit-identical output.
        for (int i = 0; i < o1.Length; i++)
            Assert.True(Math.Abs((double)o1[i] - (double)o2[i]) < 1e-6,
                $"clone output[{i}]={o2[i]} diverged from source {o1[i]} (fp16-resident clone drift, #1764)");
    }

    [Fact]
    public void GetSetParameters_UnderFp16Resident_RoundTrips()
    {
        var predictor = CreateResidentPredictor(seed: 321);
        var input = Rand(new[] { 1, 16, 8, 8 }, seed: 11);
        var conditioning = Rand(new[] { 1, 1, 64 }, seed: 13);

        // Force materialization + fp16-resident downcast.
        _ = predictor.PredictNoise(input, timestep: 0, conditioning: conditioning);

        // GetParameters must return the full parameter set (resident layers read their half masters, not the
        // freed [0,0] fp32 placeholders / shared upcast scratch) — the count must match ParameterCount.
        var flat = predictor.GetParameters();
        Assert.Equal(predictor.ParameterCount, flat.Length);

        // Round-trip: setting the same parameters back and re-running must reproduce the output exactly.
        var before = predictor.PredictNoise(input, timestep: 0, conditioning: conditioning);
        predictor.SetParameters(flat);
        var after = predictor.PredictNoise(input, timestep: 0, conditioning: conditioning);
        for (int i = 0; i < before.Length; i++)
            Assert.True(Math.Abs((double)before[i] - (double)after[i]) < 1e-6,
                $"resident GetParameters→SetParameters round-trip changed output at [{i}] (#1764)");
    }
}
