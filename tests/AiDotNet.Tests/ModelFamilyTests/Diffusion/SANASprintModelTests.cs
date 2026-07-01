using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): instantiated at full default (paper) scale — a single model's construct +
// forward/train probe peak exceeds the 16 GB PR runner, which SIGTERM-cancelled the SA-SD shard.
// Runs in the nightly HeavyTimeout lane; the default gate filters Category!=HeavyTimeout. Drop once
// it fits the per-test budget (e.g. via a tiny same-architecture injection like SDXLTurbo).
[Xunit.Collection("FoundationScaleSerial")]
[Xunit.Trait("Category", "HeavyTimeout")]
public class SANASprintModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new SANASprintModel<float>(seed: 42);
}
