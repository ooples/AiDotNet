using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout (#1706): foundation-scale video-editing diffusion — verified OOM
// (System.OutOfMemoryException under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling)
// OS-OOM-kills the Diffusion D-I shard. Nightly heavy lane; drop once streaming fits it.
[Xunit.Trait("Category", "HeavyTimeout")]
public class InstructVid2VidModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new InstructVid2VidModel<float>(seed: 42);
}
