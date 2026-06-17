using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class Magic3DModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new Magic3DModel<float>(seed: 42);
}
