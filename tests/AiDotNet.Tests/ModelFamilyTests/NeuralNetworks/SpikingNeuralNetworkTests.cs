using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SpikingNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    // A hard-spike SNN has a single-sample loss "noise floor" (~5e-3 MSE here): near
    // an optimum an infinitesimal weight change flips a hidden Heaviside spike, which
    // jumps the readout by O(W_out) irrespective of step size, so the loss cannot be
    // driven below the floor and floats within it. MoreData_ShouldNotDegrade compares a
    // 50-step run on one sample against a 200-step run on a DIFFERENT sample; both land
    // on that floor, so their difference is floor-vs-floor oscillation (~2e-3 measured),
    // not divergence — well under this bound but over the smooth-model default (1e-4).
    // Loosen the tolerance to the floor band; it still fails hard on real divergence
    // (NaN / loss → O(0.1+)), the failure mode the invariant exists to catch. The
    // single-sample Training/Memorization invariants keep the tight defaults — the
    // deterministic far-from-target init (SpikingNetworkCore, #1452) gives them a ~200x
    // reduction margin, so they pass without any tolerance change.
    protected override double MoreDataTolerance => 0.02;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SpikingNeuralNetwork<double>();
}
