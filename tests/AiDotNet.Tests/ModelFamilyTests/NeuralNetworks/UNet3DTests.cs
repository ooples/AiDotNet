using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #2087 / #1706: out of the HeavyTimeout lane. The default config is a paper-scale 3D U-Net (a
// 32x32x32 voxel grid through 4 encoder blocks, base 32 filters) whose training invariants ran past
// the 120 s gate. Per the #1706 out-of-memory ladder the test caps the grid at 16x16x16 and shrinks
// the network to 2 encoder blocks with 8 base filters; the per-voxel single-class head is unchanged.
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

    protected override int[] InputShape => [1, Voxels, Voxels, Voxels];
    protected override int[] OutputShape => [1, Voxels, Voxels, Voxels];

    private const int Voxels = 16;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UNet3D<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: Voxels,
                inputWidth: Voxels,
                inputDepth: Voxels,
                outputSize: 1),
            options: new UNet3DOptions { VoxelResolution = Voxels, NumEncoderBlocks = 2, BaseFilters = 8 });
}
