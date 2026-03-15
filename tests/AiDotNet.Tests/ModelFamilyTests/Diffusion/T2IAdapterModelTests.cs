using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class T2IAdapterModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new T2IAdapterModel<double>();
}
