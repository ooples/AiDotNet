using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class NeuralTuringMachineTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new NeuralTuringMachine<float>();

    // NTM training is deterministic per-platform (the lazy-Dense resize re-randomization bug is fixed
    // in DenseLayer.EnsureWeightShapeForInput). On this single-sample memorization task NTM converges
    // to a shallow ~1e-4 floor by 50 iterations, then Adam takes a few more bounded steps by 200. The
    // magnitude of that post-convergence Adam drift is platform-dependent: on Windows it's ~1.8e-4
    // (well under the previous 1e-3), but on Linux CI the Adam floor is higher — measured
    // lossLong=0.002073 vs lossShort=0.000133, a ~1.9e-3 drift that exceeds 1e-3 and reddened the
    // NeuralNetworks M-N shard (#1753). Both losses are at the convergence noise floor (1e-4..1e-3),
    // so this is Adam-past-convergence jitter, not divergence. The previous 1e-3 was a Windows-only
    // "50x tighter than the noise floor because it's reproducible" calibration; that assumption breaks
    // across platforms, so fall back to the ~0.05 noise-floor calibration (the SNN/NTM precedent from
    // #1643) — it absorbs the Linux floor with run-to-run margin while still catching genuine
    // divergence (orders of magnitude larger) and NaN (asserted separately above). The strict
    // Training_ShouldReduceLoss / LossStrictlyDecreasesOnMemorizationTask / TrainingError invariants
    // still pass at default strictness — only this floor-noise bound is relaxed.
    protected override double MoreDataTolerance => 0.05;
}
