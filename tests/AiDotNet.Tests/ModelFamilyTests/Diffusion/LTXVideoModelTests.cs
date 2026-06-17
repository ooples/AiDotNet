using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class LTXVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 128, 32, 32];
    protected override int[] OutputShape => [1, 128, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new LTXVideoModel<float>(seed: 42);
}
