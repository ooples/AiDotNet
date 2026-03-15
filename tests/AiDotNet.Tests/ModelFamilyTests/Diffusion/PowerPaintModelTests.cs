using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class PowerPaintModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new PowerPaintModel<double>();
}
