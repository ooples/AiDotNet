using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class FastTextTests : NeuralNetworkModelTestBase<float>
{
    protected override INeuralNetworkModel<float> CreateNetwork()
        => new FastText<float>();
}
