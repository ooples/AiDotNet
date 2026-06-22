using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class StableAudioModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 64, 32, 32];
    protected override int[] OutputShape => [1, 64, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new StableAudioModel<float>();
}
