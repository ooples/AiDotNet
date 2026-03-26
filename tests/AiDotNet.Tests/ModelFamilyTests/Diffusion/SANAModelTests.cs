using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SANAModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 32, 64, 64];
    protected override int[] OutputShape => [1, 32, 64, 64];

    protected override IDiffusionModel<double> CreateModel()
        => new SANAModel<double>(seed: 42);
}
