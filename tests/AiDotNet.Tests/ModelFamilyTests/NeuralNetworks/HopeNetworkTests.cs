using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HopeNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [256];
    protected override int[] OutputShape => [256];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HopeNetwork<double>();
}
