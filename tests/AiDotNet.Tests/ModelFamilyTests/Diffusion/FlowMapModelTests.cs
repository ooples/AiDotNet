using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (an SD 1.5-scale UNet, base 320, and a
// 128-base VAE on a 64x64 latent) the training peak ran out of memory on the nightly runner
// (run 39554054058). Per the #1706 out-of-memory ladder the test caps the latent at 16x16 and
// injects a small same-architecture UNet and VAE: the latent channels (4) and the text context
// width (768) stay paper-correct, and only width and depth shrink.
public class FlowMapModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new FlowMapModel<float>(
            predictor: new UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4,
                baseChannels: 32, channelMultipliers: [1, 2],
                numResBlocks: 1, attentionResolutions: [1],
                contextDim: 768, numHeads: 4, inputHeight: 16, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}