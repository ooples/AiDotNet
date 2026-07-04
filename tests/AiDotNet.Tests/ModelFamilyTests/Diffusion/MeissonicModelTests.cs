using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale model: Clone_ShouldProduceIdenticalOutput exceeds the 120s
// [Fact(Timeout)] gate in ISOLATION (verified solo, fresh process), so it belongs in the HeavyTimeout
// nightly lane rather than the default PR gate (#1706/#1305) - the Clone logic is correct; the model is
// simply too large to run a forward within the envelope.
[Xunit.Trait("Category", "HeavyTimeout")]
public class MeissonicModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new MeissonicModel<float>(seed: 42);
}
