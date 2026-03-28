using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VoxelCNNTests : NeuralNetworkModelTestBase
{
    // VoxelCNN default: 32x32x32 voxels, 1 channel
    // Actual output is 128-dim from conv feature extraction
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VoxelCNN<double>();
}
