using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class JEN1ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new JEN1Model<double>();
}
