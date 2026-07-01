using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale Point-E point-cloud diffusion. Verified genuine OOM — throws
// System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// runner ceiling (Metadata_ShouldExist alone OOMs), OS-OOM-killing the Diffusion N-R shard. Runs in the
// nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class PointEModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 6, 32, 32];
    protected override int[] OutputShape => [1, 6, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new PointEModel<float>(seed: 42);
}
