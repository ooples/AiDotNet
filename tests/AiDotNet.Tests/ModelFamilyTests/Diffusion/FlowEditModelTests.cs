using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale flow-matching editing model. Verified genuine OOM — throws
// System.OutOfMemoryException in the training invariants under a 16 GB DOTNET_GCHeapHardLimit reproducing
// the CI runner ceiling, OS-OOM-killing the Diffusion D-I shard. Runs in the nightly heavy lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class FlowEditModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new FlowEditModel<float>(seed: 42);
}
