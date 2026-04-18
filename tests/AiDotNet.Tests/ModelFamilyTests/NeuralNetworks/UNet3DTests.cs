using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class UNet3DTests : NeuralNetworkModelTestBase
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
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1, 32, 32, 32];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new UNet3D<double>();
}
