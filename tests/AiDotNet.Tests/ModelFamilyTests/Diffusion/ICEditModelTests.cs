using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout (#1706): foundation-scale image-editing diffusion — verified OOM
// (ForwardPass throws System.OutOfMemoryException under a 16 GB DOTNET_GCHeapHardLimit reproducing the
// CI ceiling) OS-OOM-kills the Diffusion D-I shard. Nightly heavy lane; drop once streaming fits it.
// NOTE: separately, Clone_ShouldProduceIdenticalOutput fails even at full memory (clone output diverges
// from the source) — a real ICEdit clone-fidelity bug tracked independently, NOT masked by this tag.
[Xunit.Trait("Category", "HeavyTimeout")]
public class ICEditModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ICEditModel<float>(seed: 42);
}
