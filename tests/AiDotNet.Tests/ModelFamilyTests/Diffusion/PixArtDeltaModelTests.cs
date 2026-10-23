using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (a 1152-wide, 28-layer DiT and a
// 128-base VAE) the model ran out of memory on the nightly runner (run 39554054058). Per the #1706
// out-of-memory ladder the test caps the latent at 16x16 and injects small same-architecture
// components: the latent channels (4), cross-attention width (4096) and patch size (2) stay
// paper-correct, and only width and depth shrink.
public class PixArtDeltaModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new PixArtDeltaModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 16, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
