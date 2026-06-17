using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// IDM-VTON test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. IDM-VTON's dual-encoder SD-inpainting UNet at
/// paper defaults allocates ≈12.3 GB at FP64 standalone, OOMs in the shared
/// diffusion-test process.
/// </summary>
public class IDMVTONModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new IDMVTONModel<float>(seed: 42);
}
