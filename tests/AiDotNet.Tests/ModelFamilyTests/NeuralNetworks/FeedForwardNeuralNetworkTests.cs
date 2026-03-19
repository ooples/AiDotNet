using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class FeedForwardNeuralNetworkTests : NeuralNetworkModelTestBase
{
    // FFNN default: inputSize=128, outputSize=1, 1D tensors
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new FeedForwardNeuralNetwork<double>();
}
