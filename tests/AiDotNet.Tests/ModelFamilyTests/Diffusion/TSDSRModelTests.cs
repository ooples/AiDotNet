using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TSDSRModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // Wraps a full SD UNet (320 base, [1,2,4,4]) + VAE: a single forward over the 64x64 latent exceeds
    // the 120s model-family budget. Inject a tiny same-architecture UNet + VAE — the UNet's 8 input
    // channels (4 latent + 4 SR-conditioning concat), 4 output channels and contextDim (1024) stay
    // paper-correct; only base channels / level count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new TSDSRModel<float>(
            predictor: new UNetNoisePredictor<float>(
                inputChannels: 8, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 1024, numHeads: 4,
                inputHeight: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
