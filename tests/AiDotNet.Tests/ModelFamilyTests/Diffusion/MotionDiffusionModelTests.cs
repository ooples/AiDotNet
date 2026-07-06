using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MotionDiffusionModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 263, 32, 32];
    protected override int[] OutputShape => [1, 263, 32, 32];

    // Build the SiT predictor + VAE at a REDUCED scale instead of the default DiT-XL-class SiT
    // (hiddenSize 1152 x 28 layers), whose train-then-forward loop tips over the 120s gate under shard
    // load. The small SiT (64 x 2) drops it to a few seconds. Shape-critical dims preserved
    // (inputChannels = LATENT_CHANNELS 263, the motion feature dim).
    protected override IDiffusionModel<float> CreateModel()
        => new MotionDiffusionModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.SiTPredictor<float>(
                inputChannels: 263, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 263, latentChannels: 263, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
