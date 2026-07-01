using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale flow-based video diffusion (FlowVid). Verified genuine OOM —
// throws System.OutOfMemoryException in the training invariants under a 16 GB DOTNET_GCHeapHardLimit
// reproducing the CI runner ceiling, OS-OOM-killing the Diffusion D-I shard. Runs in the nightly heavy lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class FlowVidModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new FlowVidModel<float>(seed: 42);
}
