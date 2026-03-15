using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.WorldModels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class OasisModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new OasisModel<double>();
}
