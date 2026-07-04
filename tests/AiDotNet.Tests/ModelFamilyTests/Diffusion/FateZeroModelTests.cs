using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations) that OOMs the 16 GB CI runner (fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class FateZeroModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // Seed for deterministic, reproducible weight init (matches the sibling
    // FlowVid/InstructVid2Vid tests). Without it the predictor's lazy weights fall
    // back to the process-shared RNG, so Training_ShouldReducePredictionError becomes
    // suite-position-dependent (passes in isolation, fails interleaved).
    protected override IDiffusionModel<float> CreateModel()
        => new FateZeroModel<float>(seed: 42);
}
