using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Zero123ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new Zero123Model<double>();
}
