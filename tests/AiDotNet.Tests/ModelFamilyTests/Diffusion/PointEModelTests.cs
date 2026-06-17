using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class PointEModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 6, 32, 32];
    protected override int[] OutputShape => [1, 6, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new PointEModel<float>(seed: 42);
}
