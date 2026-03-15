using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class EDiffIModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new EDiffIModel<double>();
}
