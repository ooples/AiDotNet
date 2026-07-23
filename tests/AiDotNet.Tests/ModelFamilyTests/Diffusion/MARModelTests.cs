using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MARModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // Build the SiT predictor + VAE at a REDUCED scale instead of the default (hiddenSize 1024 x 24
    // layers), whose multi-iteration Training blows the 120s gate. Shape-critical dims preserved
    // (inputChannels = LATENT_CHANNELS 16) so the forward/patchify path is exercised identically; the
    // test stays exact, fast, and in the PR gate. Mirrors OmniGen2ModelTests/SeedEdit3ModelTests.
    protected override IDiffusionModel<float> CreateModel()
        => new MARModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.SiTPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
