using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SANAModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 64, 64];
    protected override int[] OutputShape => [1, 32, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SANAModel<float>(seed: 42);
}
