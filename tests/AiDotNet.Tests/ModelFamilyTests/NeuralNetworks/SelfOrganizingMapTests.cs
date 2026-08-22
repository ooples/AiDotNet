using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SelfOrganizingMapTests : NeuralNetworkModelTestBase<float>
{
    // outputSize=64 allocates EXACTLY 64 neurons. The old grow-then-shrink heuristic settled on a
    // 10x6=60 grid and silently allocated fewer neurons than requested; SelfOrganizingMap now picks
    // the factor pair closest to the golden ratio, which for 64 is 8x8 (#1789). This expectation
    // said 60 and was stale -- it went unnoticed because Predict was returning an identity
    // passthrough of the 128-length input, so the declared output shape was never exercised.
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [64];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SelfOrganizingMap<float>();

    /// <summary>
    /// SOM uses competitive learning with one-hot BMU output.
    /// MSE tolerance is higher because output depends on which neuron wins.
    /// </summary>
    protected override double MoreDataTolerance => 0.05;
}
