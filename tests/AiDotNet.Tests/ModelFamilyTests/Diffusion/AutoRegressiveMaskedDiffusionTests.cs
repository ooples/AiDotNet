using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale autoregressive diffusion (same FastGeneration family as
// ARDiffusionModel, with a larger [1,16,32,32] state): a single training step exceeds the
// 120s [Fact(Timeout)] in isolation (verified — Predict alone ~59s, one Train step >120s),
// so the training probe cannot fit the default per-test gate. Belongs in the HeavyTimeout
// nightly lane rather than the default PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class AutoRegressiveMaskedDiffusionTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new AutoRegressiveMaskedDiffusion<float>(seed: 42);
}
