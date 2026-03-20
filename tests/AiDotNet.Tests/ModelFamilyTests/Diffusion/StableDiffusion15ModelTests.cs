using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableDiffusion15ModelTests : DiffusionModelTestBase
{
    // Per Rombach et al. 2022: SD 1.5 operates on 64×64 latent space
    // with 4 channels from the VAE (512×512 images / 8x downsampling)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // Use default constructor — all architecture parameters match the paper:
    // UNet: baseChannels=320, multipliers=[1,2,4,4], 2 ResBlocks, attention at [1,2,3]
    // VAE: baseChannels=128, multipliers=[1,2,4,4], latentScale=0.18215
    // Users can customize via constructor params, but defaults = paper values
    protected override IDiffusionModel<double> CreateModel()
        => new StableDiffusion15Model<double>(seed: 42);
}
