using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphIsomorphismNetworkTests : GraphNNModelTestBase
{
    // GIN default: inputSize=128, outputSize=7, 10 nodes
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphIsomorphismNetwork<double>();
}
