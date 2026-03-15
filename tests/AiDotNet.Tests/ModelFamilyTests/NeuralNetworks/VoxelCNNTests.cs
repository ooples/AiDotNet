using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VoxelCNNTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VoxelCNN<double>();
}
