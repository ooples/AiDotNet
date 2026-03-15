using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class FateZeroModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new FateZeroModel<double>();
}
