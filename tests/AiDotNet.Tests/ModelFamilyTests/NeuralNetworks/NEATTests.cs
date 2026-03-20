using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class NEATTests : NeuralNetworkModelTestBase
{
    // NEAT default: inputSize=10, outputSize=1
    protected override int[] InputShape => [10];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new NEAT<double>();
}
