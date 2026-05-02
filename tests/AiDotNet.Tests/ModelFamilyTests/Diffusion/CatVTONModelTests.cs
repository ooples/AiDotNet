using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class CatVTONModelTests : DiffusionModelTestBase
{
    // Paper-faithful CatVTON ships a 320-base-channel SD UNet with [1,2,4,4]
    // channel multipliers. At the production 64×64 latent (512×512 image /
    // 8× VAE) a 10-step Predict on single-threaded CPU CI exceeds the
    // [Fact] 120 s timeout. The invariant tests only check shape/finiteness
    // and clone semantics — they don't depend on full-resolution latents —
    // so use an 8×8 latent (~64× compute reduction) to fit the budget while
    // leaving the model's paper-faithful defaults untouched for production.
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<double> CreateModel()
        => new CatVTONModel<double>(seed: 42);
}
