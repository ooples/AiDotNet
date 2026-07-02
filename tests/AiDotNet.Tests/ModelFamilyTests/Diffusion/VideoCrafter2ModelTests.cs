using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale VideoCrafter2 video diffusion. Verified genuine OOM — throws
// System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// runner ceiling (Metadata_ShouldExist alone OOMs), OS-OOM-killing the Diffusion T-Z shard. Runs in the
// nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class VideoCrafter2ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new VideoCrafter2Model<float>();
}
