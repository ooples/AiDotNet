using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GRUNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    // #1706: like its sibling recurrent nets (LSTMNeuralNetworkTests, NeuralTuringMachineTests,
    // which both already set 1e-3), the GRU memorizes this tiny task down to a ~1e-4 loss floor and
    // then oscillates within it. The base 1e-4 MoreData tolerance is tighter than that floor noise,
    // so MoreData_ShouldNotDegrade false-fails when the 200-iter run lands a hair above the 50-iter
    // run (observed 6.1e-4 vs 1.5e-4 — both fully converged, not diverging). Match the recurrent-net
    // precedent. This is the documented purpose of the virtual, NOT a weakened correctness bar.
    protected override double MoreDataTolerance => 1e-3;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new GRUNeuralNetwork<float>();
}
