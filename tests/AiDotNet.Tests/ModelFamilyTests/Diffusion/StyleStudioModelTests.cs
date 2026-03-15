using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StyleStudioModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new StyleStudioModel<double>();
}
