using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout: foundation-scale diffusion (video / paper-scale); correct but a single
// forward x N-step Generate exceeds the 120 s per-test gate. Runs in the nightly lane.
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
