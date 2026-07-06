using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Compute-bound foundation-scale IP-Adapter clone path exceeds the 120s [Fact(Timeout)]
// gate on net471 in isolation, so it belongs in the HeavyTimeout nightly lane rather than
// the default PR gate (#1706/#1305). The clone logic is still covered full-fidelity there.
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class IPAdapterModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new IPAdapterModel<float>(seed: 42);
}
