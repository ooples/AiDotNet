using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MoMaskModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new MoMaskModel<double>();
}
