using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class PointEModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 6, 32, 32];
    protected override int[] OutputShape => [1, 6, 32, 32];

    // Build the point-cloud DiT predictor at a REDUCED scale instead of the default (hiddenSize 512 x
    // 12 layers), whose train-then-forward loop tips over the 120s gate under shard load. Shape-critical
    // dims preserved (inputChannels = POINTE_LATENT_CHANNELS 6, contextDim 1024, latentSpatialSize 32).
    protected override IDiffusionModel<float> CreateModel()
        => new PointEModel<float>(
            pointCloudPredictor: new AiDotNet.Diffusion.NoisePredictors.DiTNoisePredictor<float>(
                inputChannels: 6, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 1024, latentSpatialSize: 32, seed: 42),
            seed: 42);
}
