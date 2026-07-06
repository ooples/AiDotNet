using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ConvolutionalNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    // CNN default: 28x28x1 input (MNIST), 10 output classes
    protected override int[] InputShape => [1, 28, 28];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new ConvolutionalNeuralNetwork<float>();
}
