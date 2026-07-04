using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale DiT predictor (PixArt-α ~600M params): a single forward exceeds
// the 120s [Fact(Timeout)] in isolation (verified solo — Clone_ShouldProduceIdenticalOutput times
// out), so it belongs in the HeavyTimeout nightly lane rather than the default PR gate (#1706/#1305).
// The Clone logic is correct (added dit/vae ctor params so the clone gets resolved sub-models before
// ShareWeightsFrom); only the runtime is the issue.
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class PixArtModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new PixArtModel<float>(seed: 42);
}
