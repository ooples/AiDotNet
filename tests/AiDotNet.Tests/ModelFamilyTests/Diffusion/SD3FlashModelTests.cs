using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): SD3 MMDiT instantiated at full default (paper) scale — a single model's peak
// exceeds the 16 GB PR runner, which SIGTERM-cancelled the SA-SD shard. Runs in the nightly
// HeavyTimeout lane; the default gate filters Category!=HeavyTimeout. Drop once it fits the budget.
[Xunit.Collection("FoundationScaleSerial")]
[Xunit.Trait("Category", "HeavyTimeout")]
public class SD3FlashModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SD3FlashModel<float>(seed: 42);
}
