using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout: compute-bound foundation-scale autoregressive diffusion; correct but a
// single forward x N-step Generate (and the training probe) exceeds the 120 s per-test
// gate in isolation (verified), so it runs in the nightly lane rather than the default
// PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class ARDiffusionModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 8, 32, 32];
    protected override int[] OutputShape => [1, 8, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ARDiffusionModel<float>(seed: 42);
}
