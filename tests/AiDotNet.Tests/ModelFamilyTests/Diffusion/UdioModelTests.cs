using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class UdioModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new UdioModel<double>();
}
