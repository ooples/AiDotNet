using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class LiquidStateMachineTests : NeuralNetworkModelTestBase<float>
{
    protected override INeuralNetworkModel<float> CreateNetwork()
        => new LiquidStateMachine<float>();
}
