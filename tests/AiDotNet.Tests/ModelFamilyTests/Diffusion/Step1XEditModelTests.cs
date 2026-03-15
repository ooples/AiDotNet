using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Step1XEditModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new Step1XEditModel<double>();
}
