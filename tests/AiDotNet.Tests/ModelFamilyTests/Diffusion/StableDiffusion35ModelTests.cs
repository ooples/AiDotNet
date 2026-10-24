using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (an MMDiT-X Medium predictor, 1536
// wide and 24 layers, and a 128-base VAE; it timed out at 8 min) the model ran out of memory on the
// nightly runner (run 39554054058). Per the #1706 out-of-memory ladder the test caps the latent at
// 16x16 and injects small same-architecture components: the latent channels (16), context width
// (4096), patch size (2) and VAE scale factor stay paper-correct, and only width and depth shrink.
public class StableDiffusion35ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new StableDiffusion35Model<float>(
            predictor: new MMDiTXNoisePredictor<float>(
                variant: MMDiTXVariant.Medium, inputChannels: 16, patchSize: 2, contextDim: 4096,
                seed: 42, hiddenSizeOverride: 64, numLayersOverride: 2, numHeadsOverride: 4),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1,
                latentScaleFactor: 1.5305, seed: 42),
            seed: 42);
}
