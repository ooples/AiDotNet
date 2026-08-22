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

    // NEAREST-PROTOTYPE SELECTION IS DIRECTION-DOMINATED, so a 10x scale need not move the winner.
    //
    // A SOM's forward is argmin_n ||w_n - x||^2 over a codebook. Scaling x by ten preserves its
    // DIRECTION, and for a point far outside the codebook that argmin reduces to argmax_n (x . w_n)
    // -- the prototype best aligned with x, which is frequently the same neuron that won for x
    // itself. Measured here: both x and 10x select neuron 20 of 64. The forward pass ran correctly
    // and produced a valid one-hot; the selection simply did not move. Same structural reason
    // RadialBasisFunctionNetwork is exempt: a local/competitive architecture cannot be probed for
    // input-sensitivity by a global rescale.
    //
    // THIS EXEMPTION WAS ONLY SAFE TO ADD ONCE A REAL BUG BEHIND IT WAS FIXED. Before that fix,
    // PredictCore delegated to the GPU layer-chain path and, because a SOM has no layer chain,
    // returned its own 128-length input. That passthrough made this invariant PASS -- scaling an
    // identity trivially changes it -- so the test was green for the wrong reason and the failure
    // only appeared once Predict started producing a genuine one-hot. Opting out before fixing
    // would have buried the defect the invariant exists to catch.
    //
    // Input sensitivity remains covered: the determinism and training invariants perturb within the
    // codebook, where the BMU does move.
    protected override bool ScaledInputInvariantApplicable => false;
}
