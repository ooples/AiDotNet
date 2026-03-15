using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class UpscaleAVideoModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new UpscaleAVideoModel<double>();
}
