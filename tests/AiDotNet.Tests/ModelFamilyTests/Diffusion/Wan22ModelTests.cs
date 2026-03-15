using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Wan22ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new Wan22Model<double>();
}
