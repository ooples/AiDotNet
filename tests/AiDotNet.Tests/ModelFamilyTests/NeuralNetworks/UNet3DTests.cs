using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// UNet3D's default config is a paper-scale volumetric segmentation network: a 32×32×32 voxel grid
// (32,768 voxels) through 4 encoder/decoder blocks (baseFilters 32 → 64 → 128 → 256 at the
// bottleneck) with skip connections. Every block is a 3D convolution — O(voxels × C² × 27) MACs —
// and the suite's single-threaded determinism BLAS runs the whole forward+backward serially. All 7
// training invariants exceed the 120s per-test budget EVEN serialized (verified: the T-Z shard runs
// serial via $heavyShards, yet each UNet3D test still times out at 120000ms). This is inherent to a
// full 3D U-Net at paper resolution, not a regression and not shrinkable (never-shrink rule). Tag
// HeavyTimeout so the class is excluded from the default gate and runs full-fidelity in the nightly
// heavy lane (deferred, not skipped — it graduates back once 3D conv is fast enough); #1706/#1305.
[Trait("Category", "HeavyTimeout")]
public class UNet3DTests : NeuralNetworkModelTestBase<float>
{
    // UNet3D is a per-voxel segmentation network: it emits one class
    // prediction per input voxel, so the output carries the same spatial
    // dimensions as the input. The final 1x1x1 Conv3D produces
    // [numClasses, D, H, W] per sample — for the default single-class
    // config that is [1, 32, 32, 32], NOT [1] (which is what the previous
    // OutputShape claim produced). Without this correction the training
    // tests threw "Tensor shapes must match. Got [1, 32, 32, 32] and [1]"
    // when the loss tried to subtract a per-voxel prediction from a
    // scalar target.
    // Serializes in the nightly heavy lane so the 3D forward gets the whole machine uncontended.
    protected override bool RequiresHeavySerialization => true;

    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UNet3D<float>();
}
