using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale Recraft V3 image-generation model. Verified genuine OOM — throws
// System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// runner ceiling (Metadata_ShouldExist alone OOMs), OS-OOM-killing the Diffusion N-R shard. Runs in the
// nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class RecraftV3ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new RecraftV3Model<float>(seed: 42);
}
