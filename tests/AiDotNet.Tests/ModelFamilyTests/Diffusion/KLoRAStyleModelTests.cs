using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class KLoRAStyleModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion. Use a 16x16 latent (not the paper's 64x64) so the U-Net's
    // self-attention runs over 256 tokens instead of 4096, keeping the train-then-forward loop inside
    // the 120s gate under shard load.
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // Build the U-Net + VAE at a REDUCED width instead of the SD1.5-scale default (baseChannels 320 x
    // [1,2,4,4]). Shape-critical dims preserved (inputChannels = LATENT_CHANNELS 4, contextDim 768) so
    // the forward path is exercised identically; the test stays exact, fast, and in the PR gate.
    protected override IDiffusionModel<float> CreateModel()
        => new KLoRAStyleModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1,
                attentionResolutions: new[] { 1, 2 }, contextDim: 768, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
