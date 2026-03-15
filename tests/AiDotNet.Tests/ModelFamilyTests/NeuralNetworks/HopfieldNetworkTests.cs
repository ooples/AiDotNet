using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HopfieldNetworkTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HopfieldNetwork<double>();
}
