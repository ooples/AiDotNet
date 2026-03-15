using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TLoRAModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new TLoRAModel<double>();
}
