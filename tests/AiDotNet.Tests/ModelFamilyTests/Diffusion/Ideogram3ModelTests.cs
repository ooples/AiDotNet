using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout (#1706): foundation-scale text-to-image diffusion — verified OOM
// (System.OutOfMemoryException under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling)
// OS-OOM-kills the Diffusion D-I shard. Nightly heavy lane; drop once streaming fits it.
[Xunit.Trait("Category", "HeavyTimeout")]
public class Ideogram3ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new Ideogram3Model<float>(seed: 42);
}
