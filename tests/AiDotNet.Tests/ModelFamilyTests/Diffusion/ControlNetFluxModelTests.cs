using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale FLUX double-stream predictor (~12B params): a single forward
// exceeds the 120s [Fact(Timeout)] in isolation (verified solo — Clone_ShouldProduceIdenticalOutput
// times out), so it belongs in the HeavyTimeout nightly lane rather than the default PR gate
// (#1706/#1305). The Clone logic is correct (clones the resolved predictor/VAE); only runtime is slow.
[Xunit.Trait("Category", "HeavyTimeout")]
public class ControlNetFluxModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ControlNetFluxModel<float>(seed: 42);
}
