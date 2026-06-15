using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LatteModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // Latte defaults to a foundation-scale DiT (hidden 1152, 28 layers, contextDim 4096) + a
    // 128-base-channel VAE: a single forward exceeds the 120s model-family budget. Inject a tiny
    // same-architecture DiT + VAE — latentChannels (4) and contextDim (4096) stay paper-correct;
    // only width/depth/VAE base channels shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new LatteModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 32, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
