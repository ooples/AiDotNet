using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class CCSRModelTests : DiffusionModelTestBase
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<double> CreateModel()
        => new CCSRModel<double>(seed: 42);
}
