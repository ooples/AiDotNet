using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LuminaT2XModelTests : DiffusionModelTestBase<float>
{
    // LuminaT2X uses a 4-channel latent (T2X_LATENT_CHANNELS=4) and its Flag-DiT predictor is built
    // for a 32x32 latent (latentSize: 32 -> 256 patches at patch size 2). Input/output are the
    // 4-channel latent; the predictor patchifies/unpatchifies back to the same shape.
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new LuminaT2XModel<float>(seed: 42);
}
