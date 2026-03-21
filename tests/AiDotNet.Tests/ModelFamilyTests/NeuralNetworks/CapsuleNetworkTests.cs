using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CapsuleNetworkTests : NeuralNetworkModelTestBase
{
    // CapsuleNetwork per Sabour et al. (2017): 28x28x1 MNIST input, 10 classes
    protected override int[] InputShape => [1, 28, 28];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CapsuleNetwork<double>();
}
