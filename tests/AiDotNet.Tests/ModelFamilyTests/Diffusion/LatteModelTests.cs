using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LatteModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new LatteModel<double>();
}
