using AiDotNet.Interfaces;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class IDMVTONModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new IDMVTONModel<double>();
}
