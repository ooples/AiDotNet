using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SGPTTests : NeuralNetworkModelTestBase<float>
{
    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SGPT<float>();
}
