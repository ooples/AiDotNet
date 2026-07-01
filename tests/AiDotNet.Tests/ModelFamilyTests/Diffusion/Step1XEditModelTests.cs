using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout: the model-correctness bug here (the denoiser's params weren't discoverable by the
// trainable-parameter walk) is fixed in this branch, so training now runs. But the default predictor
// is DiT-XL scale (hiddenSize 1152 x 28 blocks), so a single training/forward step exceeds the 120s
// per-test budget — Training_ShouldReducePredictionError times out. This is the "correct but
// inherently slow" bucket: excluded from the default gate and tracked in the umbrella timeout issue
// (it graduates back once the foundation-forward perf work lands). It still runs in the nightly lane.
[Trait("Category", "HeavyTimeout")]
public class Step1XEditModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new Step1XEditModel<float>(seed: 42);
}
