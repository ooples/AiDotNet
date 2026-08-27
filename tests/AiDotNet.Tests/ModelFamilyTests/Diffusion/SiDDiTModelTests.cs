using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Exercise the real SiT/DiT predictor and VAE at a CI-scale width and depth. Production defaults
// remain DiT-XL/2 scale, while this fixture keeps every inherited lifecycle contract runnable in
// the normal PR lane instead of allowing clone/serialization regressions to hide in HeavyTimeout.
public class SiDDiTModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new SiDDiTModel<float>(
            predictor: new SiTPredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2,
                numHeads: 2, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
