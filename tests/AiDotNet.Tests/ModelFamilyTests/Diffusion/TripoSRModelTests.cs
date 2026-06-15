using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TripoSRModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // TripoSR defaults to a foundation-scale DiT (hidden 1024, 16 layers) over a 4-channel latent:
    // a single forward exceeds the 120s model-family budget. Inject a tiny same-architecture DiT + VAE —
    // latentChannels (4), patchSize (1) and contextDim (768) stay paper-correct; only hidden width /
    // depth / head count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new TripoSRModel<float>(
            transformer: new DiTNoisePredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 1, contextDim: 768, latentSpatialSize: 32, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42));
}
