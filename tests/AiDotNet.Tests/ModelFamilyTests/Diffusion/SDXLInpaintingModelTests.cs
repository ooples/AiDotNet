using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SDXLInpaintingModelTests : DiffusionModelTestBase
{
    // Paper-faithful SDXLInpainting ships a 320-base-channel SD UNet with
    // [1,2,4,4] channel multipliers. At the production 128×128 latent
    // (1024×1024 image / 8× VAE) a 10-step Predict on single-threaded CPU CI
    // exceeds the [Fact] 120 s timeout. The invariant tests only check
    // shape/finiteness and clone semantics — they don't depend on full-
    // resolution latents — so use an 8×8 latent to fit the budget while
    // leaving the model's paper-faithful defaults untouched for production.
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<double> CreateModel()
        => new SDXLInpaintingModel<double>(seed: 42);
}
