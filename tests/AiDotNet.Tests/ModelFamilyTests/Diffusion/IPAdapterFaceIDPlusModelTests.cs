using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class IPAdapterFaceIDPlusModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new IPAdapterFaceIDPlusModel<float>(seed: 42);
}
