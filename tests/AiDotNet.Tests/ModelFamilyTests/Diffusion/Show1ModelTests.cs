using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.LongVideo;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Show1ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new Show1Model<double>();
}
