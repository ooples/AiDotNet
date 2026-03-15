using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class InstructPix2PixModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new InstructPix2PixModel<double>();
}
