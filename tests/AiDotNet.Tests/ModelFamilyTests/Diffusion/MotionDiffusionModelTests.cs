using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MotionDiffusionModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new MotionDiffusionModel<double>();
}
