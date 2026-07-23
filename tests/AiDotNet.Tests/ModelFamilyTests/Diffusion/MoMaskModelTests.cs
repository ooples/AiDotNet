using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale motion-generation diffusion: the training probe
// exceeds the 120s [Fact(Timeout)] in isolation (verified), so it belongs in the
// HeavyTimeout nightly lane rather than the default PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class MoMaskModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new MoMaskModel<float>(seed: 42);
}
