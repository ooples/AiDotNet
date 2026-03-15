using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MVDreamModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new MVDreamModel<double>();
}
