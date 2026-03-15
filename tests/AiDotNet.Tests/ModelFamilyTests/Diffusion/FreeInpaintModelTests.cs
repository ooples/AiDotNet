using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class FreeInpaintModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new FreeInpaintModel<double>();
}
