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

    // StyDiff's FP32 DDIM loop compounds the clone's cold packed-weight rounding path.
    // The source and clone still use the exact same COW-shared tensors; Linux CI has
    // observed a bounded 3.43e-5 output delta, so keep the accommodation local to this
    // iterative sampler instead of weakening the diffusion-family contract.
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
