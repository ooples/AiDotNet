using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TurboEditModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new TurboEditModel<double>();
}
