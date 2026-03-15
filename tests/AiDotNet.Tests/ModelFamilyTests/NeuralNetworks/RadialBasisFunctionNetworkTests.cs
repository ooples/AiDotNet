using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RadialBasisFunctionNetworkTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new RadialBasisFunctionNetwork<double>();
}
