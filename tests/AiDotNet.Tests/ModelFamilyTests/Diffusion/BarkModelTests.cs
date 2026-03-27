using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class BarkModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 8, 32, 32];
    protected override int[] OutputShape => [1, 8, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new BarkModel<double>(seed: 42);
}
