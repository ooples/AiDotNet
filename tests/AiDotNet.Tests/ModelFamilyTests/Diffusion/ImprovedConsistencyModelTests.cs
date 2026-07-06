using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// ImprovedConsistency test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. Default config allocates ≈7.4 GB at FP64 standalone;
/// inside the diffusion-test shard (xunit runs many tests in the same process)
/// the cumulative LOH residual pushes it past 16 GB. FP32 brings the footprint
/// to ≈3.7 GB — safe to share the process with sibling tests.
/// </summary>
// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations) that OOMs the 16 GB CI runner (fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class ImprovedConsistencyModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new ImprovedConsistencyModel<float>(seed: 42);
}
