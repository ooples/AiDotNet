using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout: foundation-scale diffusion (video / paper-scale); correct but a single
// forward x N-step Generate exceeds the 120 s per-test gate. Runs in the nightly lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SANAModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 64, 64];
    protected override int[] OutputShape => [1, 32, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SANAModel<float>(seed: 42);
}
