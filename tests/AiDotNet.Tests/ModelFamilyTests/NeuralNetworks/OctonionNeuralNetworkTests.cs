using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class OctonionNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [64];
    protected override int[] OutputShape => [8];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new OctonionNeuralNetwork<float>();
}
