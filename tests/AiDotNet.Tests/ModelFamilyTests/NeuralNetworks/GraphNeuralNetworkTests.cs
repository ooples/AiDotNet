using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphNeuralNetworkTests : GraphNNModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphNeuralNetwork<double>();
}
