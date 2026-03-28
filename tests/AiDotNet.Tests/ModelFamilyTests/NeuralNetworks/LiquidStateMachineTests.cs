using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for Liquid State Machine per Maass et al. (2002).
/// Default: inputSize=128, outputSize=1, reservoirSize=100.
/// </summary>
public class LiquidStateMachineTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new LiquidStateMachine<double>();
}
