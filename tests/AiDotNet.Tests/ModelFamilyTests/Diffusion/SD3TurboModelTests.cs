using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SD3TurboModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<double> CreateModel()
        => new SD3TurboModel<double>(seed: 42);
}
