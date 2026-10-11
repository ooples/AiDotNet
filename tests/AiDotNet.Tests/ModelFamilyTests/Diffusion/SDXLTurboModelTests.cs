using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// #1714: back in the default gate. The HeavyTimeout deferral here was measured away: in the
// nightly lane (run 38117342722, one class per process under a managed heap cap of 70% of
// runner RAM) this class ran 13 of 13 tests green, slowest 8.9 s and no OutOfMemoryException
// -- far inside the 120 s per-test gate. Re-defer only against a fresh measurement.
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
