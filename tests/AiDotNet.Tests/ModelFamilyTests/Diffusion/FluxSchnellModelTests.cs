using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): correct but too slow for the default per-test gate (foundation-scale diffusion,
// ~100 s/forward x N-step Generate); runs in the nightly lane. Drop this trait once it fits the budget.
[Xunit.Trait("Category", "HeavyTimeout")]
public class FluxSchnellModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new FluxSchnellModel<float>(seed: 42);
}
