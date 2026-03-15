using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TokenFlowModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new TokenFlowModel<double>();
}
