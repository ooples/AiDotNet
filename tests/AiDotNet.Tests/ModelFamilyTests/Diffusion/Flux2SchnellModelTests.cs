using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale FLUX.1-schnell — the few-step sibling of the already-tagged
// Flux2 and the same ~12B-param DiT scale. Verified genuine OOM: throws System.OutOfMemoryException in
// ForwardPass under a 16 GB DOTNET_GCHeapHardLimit that reproduces the CI runner's memory ceiling, so on
// the real 16 GB runner it OS-OOM-kills the whole Diffusion D-I shard (the shard dies with no test
// output / no coverage artifact). Runs in the nightly heavy lane instead. Drop this trait once weight
// streaming lets it fit the default budget.
[Xunit.Trait("Category", "HeavyTimeout")]
public class Flux2SchnellModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new Flux2SchnellModel<float>(seed: 42);
}
