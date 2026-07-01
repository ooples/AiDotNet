using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale motion diffusion transformer. Verified genuine OOM — throws
// System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// runner ceiling (Metadata_ShouldExist alone OOMs in GC.AllocateNewArray), OS-OOM-killing the Diffusion
// J-M shard. Runs in the nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class MotionDiffuseModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new MotionDiffuseModel<float>(seed: 42);
}
