using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableAudioModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 64, 32, 32];
    protected override int[] OutputShape => [1, 64, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new StableAudioModel<double>();
}
