using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class OccupancyNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new OccupancyNeuralNetwork<double>();
}
