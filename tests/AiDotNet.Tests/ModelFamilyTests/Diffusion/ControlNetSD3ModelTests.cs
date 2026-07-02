using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale MMDiT-X predictor: a single forward exceeds the 120s
// [Fact(Timeout)] in isolation (verified solo — Predict_ShouldBeDeterministic and
// Clone_ShouldProduceIdenticalOutput both time out), so it belongs in the HeavyTimeout
// nightly lane rather than the default PR gate (#1706/#1305). The Clone logic itself is
// correct (matches the passing UNet-based ControlNet models); only the runtime is the issue.
[Xunit.Trait("Category", "HeavyTimeout")]
public class ControlNetSD3ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ControlNetSD3Model<float>(seed: 42);
}
