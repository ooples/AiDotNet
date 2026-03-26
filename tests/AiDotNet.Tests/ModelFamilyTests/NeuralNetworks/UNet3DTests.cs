using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class UNet3DTests : NeuralNetworkModelTestBase
{
    // UNet3D default: 32x32x32 voxels, 1 channel, outputSize=1
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new UNet3D<double>();
}
