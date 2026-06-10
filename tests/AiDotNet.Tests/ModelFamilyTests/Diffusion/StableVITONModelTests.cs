using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// StableVITON test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. StableVITON's SD-inpainting UNet at paper defaults
/// allocates ≈11.4 GB at FP64 standalone.
/// </summary>
public class StableVITONModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new StableVITONModel<float>(seed: 42);
}
