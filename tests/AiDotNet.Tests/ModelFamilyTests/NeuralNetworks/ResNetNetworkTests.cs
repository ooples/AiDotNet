using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ResNetNetworkTests : NeuralNetworkModelTestBase
{
    // ResNet requires 4D input [batch, channels, height, width] for convolutional layers
    // Default constructor: ResNet50 with 224x224x3 input, 1000-class output
    protected override int[] InputShape => [1, 3, 224, 224];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ResNetNetwork<double>();
}
