using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (a 2048-wide, 20-layer SiT and a
// 128-base VAE) the model ran out of memory on the nightly runner (run 39554054058). Per the #1706
// out-of-memory ladder the test caps the latent at 16x16 and injects small same-architecture
// components: the latent channels (32) and VAE scale factor stay paper-correct, and only width and
// depth shrink.
public class SANASprintModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 16, 16];
    protected override int[] OutputShape => [1, 32, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new SANASprintModel<float>(
            predictor: new SiTPredictor<float>(
                inputChannels: 32, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 32, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1,
                latentScaleFactor: 0.41407, seed: 42),
            seed: 42);
}
