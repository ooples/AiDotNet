using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CapsuleNetworkTests : NeuralNetworkModelTestBase
{
    // CapsuleNetwork default: inputSize=128, outputSize=10
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CapsuleNetwork<double>();
}
