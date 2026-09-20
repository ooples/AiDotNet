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

    // WHY THIS IS WIDER THAN THE FAMILY DEFAULT -- and what is still unexplained.
    //
    // Linux CI has seen this fixture's clone output differ from the source's: output[58] came
    // back 9.179571E-002 against 9.180864E-002, a difference of 1.293421E-005. Note which bound
    // that was weighed against -- at |expected| = 0.0918 the widened RELATIVE term contributes
    // just 1.38e-6 of the 1.137713E-005 allowance, so the ABSOLUTE tolerance is what the failure
    // was measured against. Widening the relative bound never addressed it.
    //
    // The explanation this comment used to give -- a cold packed-weight path in the clone
    // rounding differently from the source's warm one -- does not survive its own evidence.
    // Predict_ShouldBeDeterministic passed in the SAME CI process on the SAME shard, and it
    // predicts twice on ONE instance, cold then warm, matching to 12 decimals. Cold-versus-warm
    // does not move a float here.
    //
    // What IS established: the divergence does not reproduce on AVX2 hardware. Windows/x64 and a
    // 4-core AVX2 Linux container running the real CI shard (same five classes, same 65 tests,
    // same serialized heavy-shard runner config) both measure a clone difference of exactly
    // zero -- 8 solo runs and 3 full-shard runs, with the bit-identity check running and with it
    // skipped. SimdGemm.SgemmWithCachedB abandons its cached path outright when
    // Avx512Sgemm.CanUse, so an AVX-512 runner executes a kernel no machine available here can.
    //
    // This override is left in place only because the divergence is not yet explained. It is NOT
    // evidence that a difference is expected: every other diffusion fixture measures zero and so
    // should this one. If it holds at zero on CI now that the bit-identity check no longer runs
    // BETWEEN the two forwards (see DiffusionModelTestBase), delete this override rather than
    // keep it.
    protected override double CloneOutputRelativeTolerance => 1.5e-5;

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
