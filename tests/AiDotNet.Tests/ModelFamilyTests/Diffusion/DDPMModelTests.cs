using AiDotNet.Interfaces;
using AiDotNet.Diffusion;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DDPMModelTests : DiffusionModelTestBase
{
    // Per Ho et al. 2020 Table 1: CIFAR-10 images are 32×32×3
    // UNet operates directly on pixel space (no VAE)
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 3, 32, 32];

    // Default constructor creates paper-standard architecture:
    // baseChannels=128, multipliers=[1,2,2,2], 2 ResBlocks, attention at 16×16
    // Linear beta schedule 0.0001→0.02, 1000 train timesteps
    protected override IDiffusionModel<double> CreateModel()
        => new DDPMModel<double>(seed: 42);
}
