using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture UNetNoisePredictor and UNetNoisePredictor and StandardVAE. Channel
// counts, context widths, patch sizes and scale factors keep the model's defaults; only width and
// depth shrink.
public class ImagenModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 3, 16, 16];
    protected override int[] OutputShape => [1, 3, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new ImagenModel<float>(
            baseUnet: new UNetNoisePredictor<float>(
                inputChannels: 3, outputChannels: 3, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 4096, numHeads: 4, inputHeight: 16,
                seed: 42),
            superRes1Unet: new UNetNoisePredictor<float>(
                inputChannels: 3 * 2, outputChannels: 3, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 4096, numHeads: 4, inputHeight: 16,
                seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 3, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, latentScaleFactor: 1.0,
                seed: 42),
            seed: 42);
}
