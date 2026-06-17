using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MotionDiffusionModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 263, 32, 32];
    protected override int[] OutputShape => [1, 263, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new MotionDiffusionModel<float>(seed: 42);
}
