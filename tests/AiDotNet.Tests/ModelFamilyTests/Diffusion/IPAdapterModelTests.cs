using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class IPAdapterModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new IPAdapterModel<double>();
}
