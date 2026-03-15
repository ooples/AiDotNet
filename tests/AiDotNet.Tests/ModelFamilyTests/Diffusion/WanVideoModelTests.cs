using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class WanVideoModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new WanVideoModel<double>();
}
