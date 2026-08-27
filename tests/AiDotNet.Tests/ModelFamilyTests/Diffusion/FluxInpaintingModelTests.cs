using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Exercise the real FLUX Fill predictor/VAE graph at a CI-scale width and depth. The production
// defaults remain paper-scale (~12B parameters), but a single default forward exceeds the 120s
// contract-test budget and cannot provide useful clone/serialization feedback in a PR shard.
public class FluxInpaintingModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new FluxInpaintingModel<float>(
            predictor: new FluxDoubleStreamPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numJointLayers: 2,
                numSingleLayers: 2, numHeads: 2, patchSize: 2,
                contextDim: 4096, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 1.5305, seed: 42),
            seed: 42);
}
