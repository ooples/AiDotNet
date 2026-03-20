using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.WorldModels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Genie2ModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new Genie2Model<double>();
}
