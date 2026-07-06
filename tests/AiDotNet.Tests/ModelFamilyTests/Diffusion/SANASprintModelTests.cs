using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout (#1706): foundation-scale diffusion — verified OOM (System.OutOfMemoryException at
// CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling; Metadata_ShouldExist
// alone OOMs), OS-OOM-kills the Diffusion SA-SD shard. Nightly heavy lane; drop once streaming fits it.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SANASprintModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new SANASprintModel<float>(seed: 42);
}
