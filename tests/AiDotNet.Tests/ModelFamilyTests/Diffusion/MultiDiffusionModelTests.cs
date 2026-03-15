using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Panorama;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MultiDiffusionModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new MultiDiffusionModel<double>();
}
