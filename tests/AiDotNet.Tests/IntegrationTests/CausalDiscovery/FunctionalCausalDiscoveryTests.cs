using AiDotNet.CausalDiscovery.Functional;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for functional causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class FunctionalCausalDiscoveryTests
{
    private static Matrix<double> CreateSyntheticData()
    {
        int n = 50;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.1;
            data[i, 0] = x;
            data[i, 1] = 2.0 * x + 0.5;
            data[i, 2] = x + data[i, 1] * 0.3;
        }

        return new Matrix<double>(data);
    }

    private static readonly string[] FeatureNames = ["X0", "X1", "X2"];

    [Fact(Timeout = 120000)]
    public async Task ANM_FindsCausalStructure()
    {
        var algo = new ANMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CAM_FindsCausalStructure()
    {
        var algo = new CAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CAMUV_FindsCausalStructure()
    {
        var algo = new CAMUVAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task DirectLiNGAM_FindsCausalStructure()
    {
        var algo = new DirectLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task ICALiNGAM_FindsCausalStructure()
    {
        var algo = new ICALiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task VARLiNGAM_FindsCausalStructure()
    {
        var algo = new VARLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task PNL_FindsCausalStructure()
    {
        var algo = new PNLAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    /// <summary>
    /// Deterministic NONLINEAR invertible chain X0 → X1 → X2. IGCI (Janzing
    /// et al. 2012, "Information-geometric approach to inferring causal
    /// directions", §2) infers direction from the correlation between log|f′|
    /// and the input density — for a LINEAR f the derivative is constant, the
    /// criterion is exactly zero in both directions, and the method is
    /// explicitly unidentifiable (the paper's stated boundary case). The
    /// shared ramp fixture is linear, so a paper-faithful IGCI correctly
    /// reports no direction there. Smooth nonlinear monotone maps are IGCI's
    /// designed regime.
    /// </summary>
    private static Matrix<double> CreateNonlinearDeterministicData()
    {
        int n = 100;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x0 = 0.05 + 0.9 * i / (n - 1);   // uniform-ish on [0.05, 0.95]
            double x1 = x0 * x0 * x0;               // nonlinear invertible: f(x) = x^3
            double x2 = Math.Tanh(3.0 * x1);        // nonlinear invertible: g(x) = tanh(3x)

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task IGCI_FindsCausalStructure()
    {
        var algo = new IGCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateNonlinearDeterministicData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task RCD_FindsCausalStructure()
    {
        await Task.Yield();
        var algo = new RCDAlgorithm<double>();
        // RCD (like every LiNGAM-family method) requires NON-GAUSSIAN NOISE for identifiability. The
        // shared CreateSyntheticData() is a noiseless, perfectly-collinear ramp (X0, X1, X2 are all
        // exact linear functions of i), so the causal direction is genuinely UNIDENTIFIABLE — RCD now
        // correctly reports no edges on it (ConfoundingRatio returns NaN for degenerate evidence rather
        // than fabricating edges). Use a proper LiNGAM-identifiable fixture (Uniform(-1,1) noise,
        // X0 -> X1, X0 -> X2) so RCD can actually recover a structure to assert on.
        var graph = algo.DiscoverStructure(
            new Matrix<double>(BuildLingamData(200, seed: 12345, candidateIsRoot: true)), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    // ---- RCD confounding-cutoff CALIBRATION (review [1]) --------------------------------------
    // The confounding stop compares the SCALE-FREE confounding ratio
    //   Σ min(0, DiffMI)² / Σ DiffMI²  (fraction of a candidate's direction-evidence pointing the
    // WRONG way — that it is an EFFECT, not a cause) against ConfoundingEvidenceCutoff (default
    // 0.05). These tests pin the calibration: a clean root must score WELL BELOW the cutoff (so the
    // stop never fires on identifiable data — the old bug was the opposite, a dead stop) and a
    // candidate that is actually an effect / latently confounded must score WELL ABOVE it.

    // Uniform(-1,1): non-Gaussian, the identifiability condition DirectLiNGAM/RCD require.
    private static double[,] BuildLingamData(int n, int seed, bool candidateIsRoot)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        double U() => rng.NextDouble() * 2.0 - 1.0;
        var data = new double[n, 3];
        for (int k = 0; k < n; k++)
        {
            if (candidateIsRoot)
            {
                // X0 -> X1, X0 -> X2 : variable 0 is a clean root (all direction-evidence outward).
                double x0 = U();
                data[k, 0] = x0;
                data[k, 1] = 0.9 * x0 + 0.4 * U();
                data[k, 2] = 0.7 * x0 + 0.4 * U();
            }
            else
            {
                // X1 -> X0 <- X2 : variable 0 is a pure EFFECT (sink) of two independent causes,
                // the extreme of "no clean root" — every direction-evidence term points inward.
                double x1 = U();
                double x2 = U();
                data[k, 1] = x1;
                data[k, 2] = x2;
                data[k, 0] = x1 + x2 + 0.2 * U();
            }
        }
        return data;
    }

    [Fact(Timeout = 120000)]
    public async Task RCD_ConfoundingRatio_CleanRoot_IsWellBelowCutoff()
    {
        await Task.Yield();
        int n = 600;
        var data = BuildLingamData(n, seed: 12345, candidateIsRoot: true);
        double ratio = RCDAlgorithm<double>.ConfoundingRatio(data, n, candidate: 0, remaining: new[] { 0, 1, 2 });
        Assert.True(ratio < 0.05,
            $"A clean root's confounding ratio must stay well under the 0.05 cutoff so the stop does " +
            $"not fire on identifiable data; got {ratio:F4}.");
    }

    [Fact(Timeout = 120000)]
    public async Task RCD_ConfoundingRatio_PureEffect_IsWellAboveCutoff()
    {
        await Task.Yield();
        int n = 600;
        var data = BuildLingamData(n, seed: 12345, candidateIsRoot: false);
        double ratio = RCDAlgorithm<double>.ConfoundingRatio(data, n, candidate: 0, remaining: new[] { 0, 1, 2 });
        Assert.True(ratio > 0.05,
            $"A candidate that is actually an effect must score above the 0.05 cutoff so the " +
            $"confounding stop fires (it was dead code before this calibration); got {ratio:F4}.");
    }

    // Unidentifiable-structure integration test (review [201]): when the observed variables carry no
    // identifiable directional evidence, RCD must NOT fabricate directed edges — it must report the
    // configuration as confounded/indeterminate and emit an empty ordering. The noiseless,
    // perfectly-collinear CreateSyntheticData() (X0, X1, X2 are exact linear functions of a common
    // ramp) is the canonical unidentifiable case: LiNGAM/RCD require NON-GAUSSIAN NOISE, which this
    // fixture has none of. ConfoundingRatio returns NaN (indeterminate) for the best candidate's
    // degenerate evidence, so the confounding stop fires in the FIRST round (previously suppressed by
    // the `ordering.Count > 0` guard) and no edges are written. Before these fixes ConfoundingRatio
    // returned 0 (indistinguishable from a clean root) and RCD fabricated spurious edges from the
    // perfect correlation.
    [Fact(Timeout = 120000)]
    public async Task RCD_UnidentifiableData_EmitsNoDirectedEdges()
    {
        await Task.Yield();
        // Behaviour under an unidentifiable configuration: RCD emits NO directed edges rather than
        // fabricating structure from the correlation. (The per-candidate ratio thresholds — clean root
        // well below the cutoff, pure effect well above — are pinned by the RCD_ConfoundingRatio_*
        // calibration tests above; here we assert the end-to-end anti-fabrication behaviour.)
        var algo = new RCDAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.Equal(0, graph.EdgeCount);
    }

    [Fact(Timeout = 120000)]
    public async Task CCDr_FindsCausalStructure()
    {
        var algo = new CCDrAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
