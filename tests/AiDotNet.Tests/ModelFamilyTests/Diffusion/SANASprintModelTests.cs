using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SANASprintModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new SANASprintModel<float>(seed: 42);
}
