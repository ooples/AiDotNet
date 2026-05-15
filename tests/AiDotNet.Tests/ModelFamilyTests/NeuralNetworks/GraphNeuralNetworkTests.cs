using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphNeuralNetworkTests : GraphNNModelTestBase
{
    // GraphNeuralNetwork default ctor (Kipf & Welling 2017 "Semi-Supervised
    // Classification with Graph Convolutional Networks"): inputSize 128 → 7
    // classes, 10 nodes per graph.
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphNeuralNetwork<double>();
}
