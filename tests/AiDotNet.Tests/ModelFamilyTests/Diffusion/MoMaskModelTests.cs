using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MoMaskModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new MoMaskModel<float>(seed: 42);
}
