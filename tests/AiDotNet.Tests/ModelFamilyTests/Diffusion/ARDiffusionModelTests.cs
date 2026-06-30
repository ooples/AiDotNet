using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale autoregressive diffusion: the training probe
// exceeds the 120s [Fact(Timeout)] in isolation (verified), so it belongs in the
// HeavyTimeout nightly lane rather than the default PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class ARDiffusionModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 8, 32, 32];
    protected override int[] OutputShape => [1, 8, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ARDiffusionModel<float>(seed: 42);
}
