using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MeissonicModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<double> CreateModel()
        => new MeissonicModel<double>(seed: 42);
}
