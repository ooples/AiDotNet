using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class CogView4ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new CogView4Model<double>();
}
