using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// DeepFloydIF test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. DeepFloyd IF at paper defaults OOMs in fresh-process
/// probes at FP64 on the 16 GB CI host.
/// </summary>
// HeavyTimeout (#1706): after the input-channel shape fix above, DeepFloyd IF still cleanly OOMs —
// verified System.OutOfMemoryException under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling
// (foundation-scale pixel-space cascaded diffusion; the summary above notes it OOMs at paper defaults on
// the 16 GB host). So it OS-OOM-kills the Diffusion D-I shard. Runs in the nightly heavy lane; the shape
// fix keeps it correct there. Drop once weight streaming lets it fit the default budget.
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
[Xunit.Trait("Category", "HeavyTimeout")]
public class DeepFloydIFModelTests : DiffusionModelTestBase<float>
{
    // DeepFloyd IF is a PIXEL-space cascaded diffusion model operating on RGB images (3 channels),
    // not a 4-channel latent — its first conv has input depth 3, so a [1,4,64,64] probe threw
    // "Expected input depth 3, but got 4". Feed the correct 3-channel RGB shape.
    protected override int[] InputShape => [1, 3, 64, 64];
    protected override int[] OutputShape => [1, 3, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new DeepFloydIFModel<float>(seed: 42);
}
