using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableCascadeModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 24, 64, 64];
    protected override int[] OutputShape => [1, 24, 64, 64];

    protected override IDiffusionModel<double> CreateModel()
        => new StableCascadeModel<double>(seed: 42);
}
