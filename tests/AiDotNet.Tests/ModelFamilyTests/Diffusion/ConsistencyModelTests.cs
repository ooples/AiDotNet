using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations ~ 4x the ~1 GB SD/DiT-scale weights) that OOMs the 16 GB CI
// runner (verified via the CI logs — testhost/runner OOM at default scale; fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class ConsistencyModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new ConsistencyModel<float>(seed: 42);
}
