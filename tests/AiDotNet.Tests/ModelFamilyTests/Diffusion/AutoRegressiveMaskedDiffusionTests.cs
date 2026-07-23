using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale autoregressive diffusion (same FastGeneration family as
// ARDiffusionModel, with a larger [1,16,32,32] state): a single training step exceeds the
// 120s [Fact(Timeout)] in isolation (verified — Predict alone ~59s, one Train step >120s),
// so the training probe cannot fit the default per-test gate. Belongs in the HeavyTimeout
// nightly lane rather than the default PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class AutoRegressiveMaskedDiffusionTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // Build the SiT predictor + VAE at a REDUCED scale instead of the default DiT-XL-class SiT
    // (hiddenSize 1152 x 28 layers). At full scale a two-forward test runs ~54s solo and tips over the
    // 120s gate under shard load. The small SiT (64 x 2) drops it to a few seconds. Shape-critical dims
    // preserved (inputChannels = LATENT_CHANNELS 16). Mirrors MARModelTests/SeedEdit3ModelTests.
    protected override IDiffusionModel<float> CreateModel()
        => new AutoRegressiveMaskedDiffusion<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.SiTPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
