using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (an SDXL-scale inpainting UNet, base
// 320, and a 128-base VAE on a 64x64 latent) training diverged on the nightly runner (run
// 39554054058). Per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects a
// small same-architecture UNet and VAE: the 9 input channels (4 latent + 4 masked latent + 1 mask)
// and the SDXL context width (2048) stay paper-correct, and only width and depth shrink.
public class BrushNetXModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new BrushNetXModel<float>(
            predictor: new UNetNoisePredictor<float>(
                inputChannels: 9, outputChannels: 4,
                baseChannels: 32, channelMultipliers: [1, 2],
                numResBlocks: 1, attentionResolutions: [2],
                contextDim: 2048, numHeads: 4, inputHeight: 16, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
