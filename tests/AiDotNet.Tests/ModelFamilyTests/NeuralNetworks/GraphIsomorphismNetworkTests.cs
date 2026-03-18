using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphIsomorphismNetworkTests : GraphNNModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphIsomorphismNetwork<double>();
}
