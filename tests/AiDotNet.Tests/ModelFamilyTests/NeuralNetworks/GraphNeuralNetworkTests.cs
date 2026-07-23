using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphNeuralNetworkTests : GraphNNModelTestBase<float>
{
    // GraphNeuralNetwork default ctor (Kipf & Welling 2017 "Semi-Supervised
    // Classification with Graph Convolutional Networks"): Cora's 1433 input
    // features → 7 classes, 10 nodes in this synthetic graph.
    protected override int[] InputShape => [10, 1433];
    protected override int[] OutputShape => [10, 7];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new GraphNeuralNetwork<float>();
}
