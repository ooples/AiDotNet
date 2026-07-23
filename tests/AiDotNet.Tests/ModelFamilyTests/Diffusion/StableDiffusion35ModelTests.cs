using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale Stable Diffusion 3.5 MM-DiT — verified OOM
// (System.OutOfMemoryException at CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// ceiling; Metadata_ShouldExist alone OOMs), OS-OOM-kills the Diffusion Stable shard. Nightly heavy lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class StableDiffusion35ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new StableDiffusion35Model<float>(seed: 42);
}
