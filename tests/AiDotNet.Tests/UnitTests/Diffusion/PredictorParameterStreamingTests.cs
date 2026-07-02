using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Verifies the #1624 streaming parameter API (GetParameterChunks / SetParameterChunks) on the
/// foundation-scale DiT-family predictors using TINY / small variants so the invariants are checked
/// without the multi-billion-parameter allocation that OOMs at default size. The load-bearing
/// contract is PER-INDEX correspondence: the flat concatenation of GetParameterChunks must equal
/// GetParameters element-for-element (the optimizer reconstructs per-tensor gradient deltas in chunk
/// order), and SetParameterChunks must apply a streamed source back exactly.
/// </summary>
public class PredictorParameterStreamingTests
{
    private static FlagDiTPredictor<double> FlagDiT(int seed) =>
        new FlagDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2,
            numHeads: 4, numKVHeads: 2, contextDim: 32, latentSize: 8, seed: seed);

    private static AsymmDiTPredictor<double> AsymmDiT(int seed) =>
        new AsymmDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4, contextDim: 32, seed: seed);

    private static SiTPredictor<double> SiT(int seed) =>
        new SiTPredictor<double>(inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4, seed: seed);

    // EMMDiT has FIXED internal dims (1024 hidden / 12 joint blocks / 16 heads) and no size override, so
    // it cannot be scaled down — it is genuinely ~540 M parameters (NOT the ~15M an earlier comment
    // claimed). At FP64 that is ~4.3 GB resident PER instance, and SetChunks round-trips TWO instances
    // plus two flat vectors (~17 GB) — over the 16 GB CI runner. Use FP32 (the production-canonical
    // precision, matching the foundation-scale round-trip tests in FastGenContractTests), which halves
    // the footprint to ~8.6 GB so the contract is exercised without OOM. The chunk-framing contract is
    // precision-independent (pure read/copy, no arithmetic), so FP32 is a faithful check, not a weakening.
    private static EMMDiTPredictor<float> EMMDiT(int seed) =>
        new EMMDiTPredictor<float>(inputChannels: 4, contextDim: 64, seed: seed);

    // MMDiT-X exposes size overrides for a reduced-scale same-architecture fixture; it also has a raw
    // positional-embedding table appended after the layers, so this exercises the mixed layer + raw-array
    // chunk path.
    private static MMDiTXNoisePredictor<double> MMDiTX(int seed) =>
        new MMDiTXNoisePredictor<double>(
            MMDiTXVariant.Medium, inputChannels: 4, patchSize: 2, contextDim: 32, seed: seed,
            hiddenSizeOverride: 32, numLayersOverride: 2, numHeadsOverride: 4);

    // MMDiT has the deepest per-block sub-layer fan-out (18 layers / joint block, 8 / single block);
    // include one of each so the chunk sequence covers both block kinds and the head/tail layers.
    private static MMDiTNoisePredictor<double> MMDiT(int seed) =>
        new MMDiTNoisePredictor<double>(
            inputChannels: 4, hiddenSize: 32, numJointLayers: 1, numSingleLayers: 1,
            numHeads: 4, patchSize: 2, contextDim: 32, seed: seed);

    [Fact] public void FlagDiT_Chunks_IndexIdentical() => AssertIndexIdentical(FlagDiT(7));
    [Fact] public void FlagDiT_SetChunks_RoundTrips() => AssertRoundTrips(FlagDiT(1), FlagDiT(2));

    [Fact] public void AsymmDiT_Chunks_IndexIdentical() => AssertIndexIdentical(AsymmDiT(7));
    [Fact] public void AsymmDiT_SetChunks_RoundTrips() => AssertRoundTrips(AsymmDiT(1), AsymmDiT(2));

    [Fact] public void SiT_Chunks_IndexIdentical() => AssertIndexIdentical(SiT(7));
    [Fact] public void SiT_SetChunks_RoundTrips() => AssertRoundTrips(SiT(1), SiT(2));

    // HeavyTimeout: EMMDiT has FIXED foundation-scale dims (~540 M params, no size override), so even at
    // FP32 the round-trip's two instances + flat vectors sit near the 16 GB runner ceiling and runtime.
    // Keep the true-scale coverage but route it to the nightly HeavyTimeout lane so the default PR gate
    // (Category!=HeavyTimeout) stays fast and stable; the tiny FlagDiT/AsymmDiT/SiT/MMDiT(X) fixtures
    // already exercise the same chunk-framing code paths on every PR run.
    [Trait("Category", "HeavyTimeout")]
    [Fact] public void EMMDiT_Chunks_IndexIdentical() => AssertIndexIdentical(EMMDiT(7));
    [Trait("Category", "HeavyTimeout")]
    [Fact] public void EMMDiT_SetChunks_RoundTrips() => AssertRoundTrips(EMMDiT(1), EMMDiT(2));

    [Fact] public void MMDiTX_Chunks_IndexIdentical() => AssertIndexIdentical(MMDiTX(7));
    [Fact] public void MMDiTX_SetChunks_RoundTrips() => AssertRoundTrips(MMDiTX(1), MMDiTX(2));

    [Fact] public void MMDiT_Chunks_IndexIdentical() => AssertIndexIdentical(MMDiT(7));
    [Fact] public void MMDiT_SetChunks_RoundTrips() => AssertRoundTrips(MMDiT(1), MMDiT(2));

    // U-ViT: encoder/middle/decoder blocks with optional-null sub-layers + interleaved skip projections.
    // Its block attention layers only allocate weights on Forward, so (like MMDiTX) the real round-trip
    // path is Clone (which probe-forwards first). We resolve via a probe forward before the chunk set,
    // mirroring that contract. This also regression-covers the previously-truncated SetParameters and the
    // Clone that didn't materialize the clone before copying.
    private static UViTNoisePredictor<double> UViT(int seed) =>
        new UViTNoisePredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4,
            patchSize: 2, contextDim: 0, latentSpatialSize: 8, seed: seed);

    private static Tensor<double> UViTInput()
    {
        var t = new Tensor<double>(new[] { 1, 4, 8, 8 });
        for (int i = 0; i < t.Length; i++) t[i] = (i % 7) * 0.01 - 0.03;
        return t;
    }

    [Fact]
    public void UViT_Chunks_IndexIdentical()
    {
        var p = UViT(7);
        p.PredictNoise(UViTInput(), 0); // resolve lazy attention weights
        AssertIndexIdentical(p);
    }

    [Fact]
    public void UViT_SetChunks_RoundTrips()
    {
        var src = UViT(1); var dst = UViT(2);
        src.PredictNoise(UViTInput(), 0);
        dst.PredictNoise(UViTInput(), 0);
        AssertRoundTrips(src, dst);
    }

    [Fact]
    public void UViT_Clone_RoundTripsEveryLayer()
    {
        var src = UViT(1);
        src.PredictNoise(UViTInput(), 0);
        var sf = src.GetParameters();

        var clone = (UViTNoisePredictor<double>)src.Clone();
        var cf = clone.GetParameters();

        Assert.Equal(sf.Length, cf.Length);
        for (int i = 0; i < sf.Length; i++)
            Assert.Equal(sf[i], cf[i], 12);
    }

    // Generic over the element type so fixtures can choose precision (FP64 for the tiny ones; FP32 for the
    // ~540 M EMMDiT so two instances fit the 16 GB runner). Values are compared after widening to double:
    // for an FP64 predictor this is identical to the previous Assert.Equal(.., 12) behavior, and for FP32
    // the chunk path reads the SAME stored values as the flat path (no arithmetic), so they widen
    // bit-identically — 12 decimal places is satisfied exactly.
    private static void AssertIndexIdentical<T>(NoisePredictorBase<T> predictor) where T : struct
    {
        var flat = predictor.GetParameters();

        // Stream each chunk's Data.Span directly against `flat` with a running offset instead of
        // buffering a second full copy (a List<T> + per-chunk ToVector()). For the ~540 M-parameter
        // EMMDiT that second copy is another multi-GB allocation on the 16 GB runner; comparing
        // in-place keeps the peak at one flat vector + one resident chunk.
        long sum = 0;
        int offset = 0;
        foreach (var chunk in predictor.GetParameterChunks())
        {
            var span = chunk.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                Assert.True(offset < flat.Length,
                    "GetParameterChunks streamed more elements than GetParameters exposes.");
                Assert.Equal(Convert.ToDouble((object)flat[offset]), Convert.ToDouble((object)span[i]), 12);
                offset++;
            }
            sum += chunk.Length;
        }

        Assert.Equal(predictor.ParameterCount, sum);
        Assert.Equal(flat.Length, offset);
    }

    private static void AssertRoundTrips<T>(NoisePredictorBase<T> source, NoisePredictorBase<T> dest) where T : struct
    {
        var sourceFlat = source.GetParameters();
        var destBefore = dest.GetParameters();

        bool anyDifferent = false;
        for (int i = 0; i < sourceFlat.Length && !anyDifferent; i++)
            if (!sourceFlat[i].Equals(destBefore[i])) anyDifferent = true;
        Assert.True(anyDifferent, "Test setup invalid: the two seeds produced identical weights.");

        dest.SetParameterChunks(source.GetParameterChunks());

        var destAfter = dest.GetParameters();
        Assert.Equal(sourceFlat.Length, destAfter.Length);
        for (int i = 0; i < sourceFlat.Length; i++)
            Assert.Equal(Convert.ToDouble((object)sourceFlat[i]), Convert.ToDouble((object)destAfter[i]), 12);
    }
}
