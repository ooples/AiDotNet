using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class OctonionNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [64];
    protected override int[] OutputShape => [8];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new OctonionNeuralNetwork<double>();
}
