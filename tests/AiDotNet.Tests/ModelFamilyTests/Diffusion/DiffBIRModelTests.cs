using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DiffBIRModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new DiffBIRModel<double>();
}
