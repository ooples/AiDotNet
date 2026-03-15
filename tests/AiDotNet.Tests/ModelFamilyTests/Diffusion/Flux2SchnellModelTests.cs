using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Flux2SchnellModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new Flux2SchnellModel<double>();
}
