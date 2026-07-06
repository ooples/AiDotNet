using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.AudioVisual;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale audio-visual video diffusion (Emu Video 2). Verified genuine
// OOM: throws System.OutOfMemoryException in ForwardPass under a 16 GB DOTNET_GCHeapHardLimit that
// reproduces the CI runner ceiling, so it OS-OOM-kills the Diffusion D-I shard (no test output). Runs
// in the nightly heavy lane. Drop once weight streaming lets it fit the default budget.
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
[Xunit.Trait("Category", "HeavyTimeout")]
public class EmuVideo2ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new EmuVideo2Model<float>(seed: 42);
}
