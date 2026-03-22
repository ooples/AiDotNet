using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SANASprintModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new SANASprintModel<double>(seed: 42);
}
