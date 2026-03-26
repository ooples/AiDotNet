using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class PointEModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 6, 32, 32];
    protected override int[] OutputShape => [1, 6, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new PointEModel<double>(seed: 42);
}
