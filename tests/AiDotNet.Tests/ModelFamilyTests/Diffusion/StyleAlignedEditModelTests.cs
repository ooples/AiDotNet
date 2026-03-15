using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StyleAlignedEditModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new StyleAlignedEditModel<double>();
}
