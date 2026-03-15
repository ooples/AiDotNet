using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StepVideoModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new StepVideoModel<double>();
}
