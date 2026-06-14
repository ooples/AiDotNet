using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class VoiceCraftModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 8, 32, 32];
    protected override int[] OutputShape => [1, 8, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new VoiceCraftModel<float>(seed: 42);
}
