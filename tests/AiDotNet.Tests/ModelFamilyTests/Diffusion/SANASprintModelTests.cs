using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SANASprintModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new SANASprintModel<double>();
}
