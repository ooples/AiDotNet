using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class IPAdapterPlusModelTests : DiffusionModelTestBase
{
    // Paper-faithful IPAdapter ships a 320-base-channel SD UNet with
    // [1,2,4,4] channel multipliers. At the production 64×64 latent (512×512
    // image / 8× VAE) a 10-step Predict on single-threaded CPU CI exceeds
    // the [Fact] 120 s timeout. The invariant tests here only check
    // shape/finiteness/clone semantics — none of them depend on a full-
    // resolution latent — so test against an 8×8 latent (compute scales
    // ∝ H×W, ~64× reduction) to fit the budget while leaving the model's
    // paper-faithful defaults untouched for production use.
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<double> CreateModel()
        => new IPAdapterPlusModel<double>(seed: 42);
}
