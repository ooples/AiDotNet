using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class CATDMModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new CATDMModel<double>();
}
