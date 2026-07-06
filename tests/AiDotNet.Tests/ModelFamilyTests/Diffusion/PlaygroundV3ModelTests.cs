using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale model: the training probe exceeds the 120s
// [Fact(Timeout)] in isolation (verified), so it belongs in the HeavyTimeout
// nightly lane rather than the default PR gate (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class PlaygroundV3ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new PlaygroundV3Model<float>(seed: 42);
}
