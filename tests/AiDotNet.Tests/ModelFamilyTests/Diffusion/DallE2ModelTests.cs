using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DallE2ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new DallE2Model<double>();
}
