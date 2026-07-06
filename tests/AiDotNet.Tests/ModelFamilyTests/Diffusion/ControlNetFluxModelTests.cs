using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale ControlNet over a FLUX (~12B-param) backbone. Verified genuine
// OOM — throws System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit
// reproducing the CI runner ceiling (Metadata_ShouldExist alone OOMs), OS-OOM-killing the Diffusion A-C
// shard with no test output. Runs in the nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class ControlNetFluxModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new ControlNetFluxModel<float>(seed: 42);
}
