using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class VideoP2PModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new VideoP2PModel<double>();
}
