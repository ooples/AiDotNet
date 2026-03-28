using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CapsuleNetworkTests : NeuralNetworkModelTestBase
{
    // CapsuleNetwork per Sabour et al. (2017): 28x28x1 MNIST input
    // Output is flattened feature map (784), not class probabilities
    protected override int[] InputShape => [1, 28, 28];
    protected override int[] OutputShape => [784];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CapsuleNetwork<double>();
}
