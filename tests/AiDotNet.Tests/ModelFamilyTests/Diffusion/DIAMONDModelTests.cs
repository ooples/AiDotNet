using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.WorldModels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale diffusion world model (DIAMOND). Its training invariants OOM —
// verified System.OutOfMemoryException in Training_ShouldReducePredictionError / *_AfterTraining /
// Metadata-after-train under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI ceiling, OS-OOM-killing
// the Diffusion D-I shard — so it must run in the nightly heavy lane. NOTE: it also has a separate,
// non-memory Clone_ShouldProduceIdenticalOutput divergence (~8% value diff, not float noise) that the
// nightly lane will still surface; tracked as #1764 (real clone bug, deferral here is forced by
// the OOM, not a way to hide it).
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
[Xunit.Trait("Category", "HeavyTimeout")]
public class DIAMONDModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new DIAMONDModel<float>();
}
