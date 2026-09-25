using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class StyDiffModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion. Use a 16x16 latent (not the paper's 64x64) so the U-Net's
    // self-attention runs over 256 tokens instead of 4096 — the multi-iteration Training loop then
    // finishes inside the 120s gate rather than timing out at the SD1.5-scale default.
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // This fixture used to override CloneOutputRelativeTolerance to 1.5e-5. Removed on the
    // condition the override itself set: that the clone difference holds at zero once the
    // bit-identity check no longer runs BETWEEN the two forwards (see DiffusionModelTestBase).
    //
    // Validation status. Clone_ShouldProduceIdenticalOutput passed in 5.52s on Linux CI with the
    // default tolerance restored -- shard "ModelFamily - Diffusion Step-Sync", 65/65 -- and passes
    // on Windows/x64 and on AVX2 Linux. That is one green CI observation, not a proof the
    // divergence is gone: the ISA of that runner was not recorded, because the numeric-environment
    // reading only printed on failure. It is emitted unconditionally from here on, so the next run
    // will say. Every other diffusion fixture measures zero and so does this one.
    //
    // If a clone divergence returns here, it is NOT evidence that a difference is expected. The
    // history, for whoever picks it up: Linux CI once saw output[58] come back 9.179571E-002
    // against 9.180864E-002, a difference of 1.293421E-005 -- and at |expected| = 0.0918 the
    // widened RELATIVE term contributed just 1.38e-6 of the 1.137713E-005 allowance, so the
    // failure was measured against the ABSOLUTE bound and the override never addressed it. The
    // "cold packed-weight path" story is contradicted by Predict_ShouldBeDeterministic, which
    // makes exactly that cold-then-warm comparison on one instance and matches to 12 decimals.
    // The divergence never reproduced on AVX2: Windows/x64 and a 4-core AVX2 Linux container
    // running the real CI shard both measured exactly zero over 8 solo and 3 full-shard runs.
    // SimdGemm.SgemmWithCachedB abandons its cached path outright when Avx512Sgemm.CanUse, so an
    // AVX-512 runner executes a kernel no machine available here can.

    // Build the U-Net + VAE at a REDUCED width instead of the SD1.5-scale default (baseChannels 320 x
    // [1,2,4,4]), which peaks ~49 GB and blows the gate. Shape-critical dims preserved (inputChannels =
    // LATENT_CHANNELS 4, contextDim 768) so the forward path is exercised identically; the test stays
    // exact, fast, and in the default PR gate.
    protected override IDiffusionModel<float> CreateModel()
        => new StyDiffModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1,
                attentionResolutions: new[] { 1, 2 }, contextDim: 768, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
