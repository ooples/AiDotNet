using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout (#1706): foundation-scale SD3 MM-DiT — verified OOM (System.OutOfMemoryException at
// CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling; Metadata_ShouldExist
// alone OOMs), OS-OOM-kills its Diffusion shard. Nightly heavy lane; drop once streaming fits it.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SD3TurboModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SD3TurboModel<float>(seed: 42);
}
