using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class ShapEModelTests : DiffusionModelTestBase<float>
{
    // Shap-E's latent is a 1024-dim shape vector (SHAPE_LATENT_DIM), i.e. a single 1024-channel token,
    // not a [4,32,32] image latent. The canonical DiT input is therefore [1, 1024, 1, 1].
    protected override int[] InputShape => [1, 1024, 1, 1];
    protected override int[] OutputShape => [1, 1024, 1, 1];

    // Shap-E defaults to a foundation-scale latent DiT (hidden 768, 16 layers): a single forward exceeds
    // the 120s model-family budget. Inject a tiny same-architecture DiT — the 1024-d shape latent
    // (inputChannels), patchSize (1) and contextDim (1024) stay paper-correct; only hidden width / depth /
    // head count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new ShapEModel<float>(
            latentPredictor: new DiTNoisePredictor<float>(
                inputChannels: 1024, hiddenSize: 128, numLayers: 2, numHeads: 4,
                patchSize: 1, contextDim: 1024, latentSpatialSize: 1, seed: 42));
}
