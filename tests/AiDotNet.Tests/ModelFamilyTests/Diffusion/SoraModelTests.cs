using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout: foundation-scale diffusion (video / paper-scale). Correct, but a single forward x
// N-step Generate (and/or the full-scale Training peak: weights + grads + Adam + activations) exceeds
// the 120 s / 16 GB PR-gate envelope, so it runs in the HeavyTimeout nightly lane (#1706/#1305/#1622).
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class SoraModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new SoraModel<float>(seed: 42);
}
