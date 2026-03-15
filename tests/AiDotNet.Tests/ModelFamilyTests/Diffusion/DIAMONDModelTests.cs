using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.WorldModels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DIAMONDModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new DIAMONDModel<double>();
}
