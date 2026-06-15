using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MultiStepConsistencyModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // Wraps a full SD1.5 UNet (320 base, [1,2,4,4]) + VAE: a single forward over the 64x64 latent
    // exceeds the 120s model-family budget. Inject a tiny same-architecture UNet + VAE — latentChannels
    // (4) and contextDim (768, SD1.5) stay paper-correct; only base channels / level count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new MultiStepConsistencyModel<float>(
            predictor: new UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 768, numHeads: 4,
                inputHeight: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
