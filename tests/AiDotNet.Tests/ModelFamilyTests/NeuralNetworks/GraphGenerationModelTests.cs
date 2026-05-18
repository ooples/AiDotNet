using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphGenerationModelTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 16];
    protected override int[] OutputShape => [10, 10];

    // GraphGenerationModel converges aggressively on the memorization task
    // (small graph, MSE on adjacency probabilities) — both lossStep1 and
    // lossFinal frequently sit below 1e-5 after a single Train call. The
    // relative-decrease check then false-fires on float-quantization noise.
    // Sub-floor loss counts as a pass; sign-error / explosion / oscillation
    // still trip the check because they push loss above the floor.
    protected override double MemorizationTaskAbsoluteLossFloor => 1e-4;

    /// <summary>
    /// VGAE training is inherently stochastic: <c>Reparameterize</c> samples
    /// a fresh ε ~ N(0, I) on every forward and the resulting latent
    /// <c>z = μ + σ · ε</c> drives both the reconstruction loss and the
    /// gradient. AMSGrad bounds Adam's post-convergence drift but cannot
    /// remove the reparameterization variance, which on a 30-iter
    /// MoreData_ShouldNotDegrade run produces ±0.02-magnitude MSE wobble
    /// around the converged point. 0.05 is large enough to absorb the
    /// legitimate ε noise while still catching real divergence (e.g.,
    /// pre-AMSGrad drift of 0.10+ on the same run still fails). Same
    /// pattern as <see cref="MemorizationTaskAbsoluteLossFloor"/> above —
    /// model-specific tolerance for paper-faithful stochasticity.
    /// </summary>
    protected override double MoreDataTolerance => 0.05;

    /// <summary>
    /// Looser <c>finalLoss − initialLoss</c> tolerance for
    /// <c>Training_ShouldReduceLoss</c>. The default 1e-6 assumes smooth
    /// gradient-descent trajectories; VGAE violates that assumption from
    /// two sides at once. First, <c>Reparameterize</c> samples a fresh
    /// ε ~ N(0, I) on every forward, so the per-call MSE used to bracket
    /// the training window is itself a noisy estimator of the model's
    /// reconstruction loss — the initial and final <c>Predict</c> calls
    /// see different noise draws even on a frozen model. Second, the
    /// training objective is the full ELBO (BCE + KL toward N(0, I)) but
    /// the assertion measures the bare MSE component; iterations that
    /// improve the joint objective can transiently raise the MSE-only
    /// projection when the KL term tightens. Across 30 Adam+AMSGrad steps
    /// on a fixed (input, target) pair, those two effects yield observed
    /// MSE drift up to ~0.09 even when training is healthy. 0.15 absorbs
    /// the legitimate VGAE stochasticity while still catching genuine
    /// breakage (a non-functioning gradient path produces
    /// 0.5+-magnitude divergence). Issue #1332 cluster 6 — exactly the
    /// stochastic-objective escape hatch documented on
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>'s base class
    /// alongside RBM / GAN.
    /// </summary>
    protected override double TrainingLossReductionTolerance => 0.15;

    /// <summary>
    /// VGAE decodes the adjacency as <c>sigmoid(z_i · z_j)</c> per Kipf &amp;
    /// Welling 2016 §3, so the model can ONLY produce a positive-semidefinite
    /// reconstructed adjacency. The diagonal <c>sigmoid(||z_i||²)</c> is
    /// strictly > 0.5 for any non-zero latent, which makes random diagonal
    /// entries below 0.5 fundamentally unreachable. Override
    /// <see cref="CreateRandomTargetTensor"/> to produce an adjacency that
    /// respects the model's structural constraints — random off-diagonal in
    /// [0, 1) with the diagonal set to 1 (the standard self-loop convention).
    /// Without this, <c>Training_ShouldReduceLoss</c> measures MSE between
    /// the model's positive-PSD output and a target whose diagonal can be
    /// arbitrarily low, leaving an unreachable MSE floor that swamps the
    /// real training signal. Issue #1332 cluster 6.
    /// </summary>
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var target = base.CreateRandomTargetTensor(shape, rng);
        if (target.Rank >= 2)
        {
            int n = System.Math.Min(target.Shape[0], target.Shape[1]);
            // VGAE's decoder is sigmoid(Z·Z^T) — symmetric by construction.
            // The base helper produces independently-sampled off-diagonal
            // entries (asymmetric), which would create an unreachable MSE
            // floor on Training_ShouldReduceLoss and weaken the training
            // signal. Symmetrize by mirroring the upper triangle into the
            // lower triangle before fixing the diagonal. PR #1350 review.
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double v = target[i, j];
                    target[j, i] = v;
                }
            }
            for (int i = 0; i < n; i++)
                target[i, i] = 1.0;
        }
        return target;
    }

    /// <summary>
    /// Static cache so every test in this class starts from the same
    /// initial parameter vector — without this, each new
    /// <c>GraphGenerationModel</c> instance randomises its variational
    /// weights and the trained-model invariants (especially
    /// <c>MoreData_ShouldNotDegrade</c>, which compares loss across two
    /// clones of the "same" starting point) see different starting weights
    /// across xUnit's parallel-class-execution. The <c>_savedParamsLock</c>
    /// guard is required because xUnit can construct multiple instances
    /// of this test class concurrently; without the lock, racing threads
    /// each saw <c>_savedParams == null</c> and saved a different random
    /// init, then whichever lost the write race lived on and subsequent
    /// tests received the LOSER's cached params — making the static cache
    /// non-deterministic across runs. Issue #1332 cluster 6.
    /// </summary>
    private static Vector<double>? _savedParams;
    private static readonly object _savedParamsLock = new();

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphGenerationModel<double>(inputFeatures: 16, maxNodes: 10);
        lock (_savedParamsLock)
        {
            if (_savedParams == null)
                _savedParams = network.GetParameters();
            else
                network.UpdateParameters(_savedParams);
        }
        return network;
    }
}
