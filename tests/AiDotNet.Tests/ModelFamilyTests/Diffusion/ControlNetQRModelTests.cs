using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class ControlNetQRModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new ControlNetQRModel<double>();
}
