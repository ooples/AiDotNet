using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class VoiceCraftModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 8, 32, 32];
    protected override int[] OutputShape => [1, 8, 32, 32];

    // VoiceCraft defaults to a foundation-scale DiT (hidden 2048, 16 layers) over an 8-channel latent:
    // a single forward exceeds the 120s model-family budget. Inject a tiny same-architecture DiT —
    // latentChannels (8), patchSize (1) and contextDim (768) stay paper-correct; only hidden width /
    // depth / head count shrink. The AudioVAE (cheap dense projections) keeps its default scale.
    protected override IDiffusionModel<float> CreateModel()
        => new VoiceCraftModel<float>(
            transformer: new DiTNoisePredictor<float>(
                inputChannels: 8, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 1, contextDim: 768, latentSpatialSize: 32, seed: 42),
            seed: 42);
}
