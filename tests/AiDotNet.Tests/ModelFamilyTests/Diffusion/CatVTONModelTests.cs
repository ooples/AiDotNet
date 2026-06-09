using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// CatVTON test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. CatVTON's SD-inpainting UNet at paper defaults
/// allocates ≈11.8 GB at FP64 standalone, OOMs in the shared diffusion-test
/// process. FP32 halves the footprint.
/// </summary>
public class CatVTONModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new CatVTONModel<float>(seed: 42);
}
