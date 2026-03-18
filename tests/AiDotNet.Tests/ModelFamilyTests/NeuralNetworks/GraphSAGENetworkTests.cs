using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphSAGENetworkTests : GraphNNModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphSAGENetwork<double>();
}
