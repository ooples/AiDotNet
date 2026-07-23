using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): correct but too slow for the default per-test gate (foundation-scale diffusion,
// ~100 s/forward x N-step Generate); runs in the nightly lane. Drop this trait once it fits the budget.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SDXLTurboModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // SDXL-Turbo wraps a full SDXL UNet (320 base, [1,2,4]) + VAE: a single forward over the 64x64
    // latent exceeds the 120s model-family budget. Inject a tiny same-architecture UNet + VAE —
    // latentChannels (4) and contextDim (2048, SDXL) stay paper-correct; only base channels / level
    // count / res-blocks shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new SDXLTurboModel<float>(
            predictor: new UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 2048, numHeads: 4,
                inputHeight: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
