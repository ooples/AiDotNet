using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class RAPHAELModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new RAPHAELModel<double>();
}
