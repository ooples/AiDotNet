using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MeshCNNTests : NeuralNetworkModelTestBase<float>
{
    // Default MeshCNN: InputFeatures=5, NumClasses=40, PoolTargets=[1800,1350,600]
    // Need enough edges to satisfy the first pool target (>1800)
    private const int NumEdges = 2000;

    protected override int[] InputShape => [NumEdges, 5];
    protected override int[] OutputShape => [40];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var mesh = new MeshCNN<float>();
        mesh.SetEdgeAdjacency(CreateTestEdgeAdjacency());
        return mesh;
    }

    // MeshCNN's default architecture carries Dropout (per Hanocka et al. 2019). On the
    // tiny fixed-sample memorization task the loss converges to a ~0.31 plateau by the
    // short (50-iter) run, after which the long (200-iter) run oscillates within the
    // Dropout + Adam-past-convergence noise band (~4e-4 drift observed) rather than
    // strictly decreasing — not divergence (LossStrictlyDecreases confirms it decreases).
    //
    // Tolerance calibrated at 1e-3: ~2.5× the observed ~4e-4 stochastic drift band,
    // enough headroom to swallow Dropout-mask/Adam-past-convergence noise without also
    // swallowing genuine regressions. Was 5e-3 (12.5× headroom), which could let a real
    // training regression pass. LossStrictlyDecreases still catches divergence from the
    // other direction, so this is the "no significant increase" complement.
    protected override double MoreDataTolerance => 1e-3;

    /// <summary>
    /// Creates a simple circular edge adjacency matrix for testing.
    /// Each edge has 4 neighbors (default NumNeighbors) arranged in a ring topology.
    /// </summary>
    private static int[,] CreateTestEdgeAdjacency()
    {
        var adjacency = new int[NumEdges, 4];
        for (int i = 0; i < NumEdges; i++)
        {
            adjacency[i, 0] = (i + 1) % NumEdges;
            adjacency[i, 1] = (i + 2) % NumEdges;
            adjacency[i, 2] = (i + NumEdges - 1) % NumEdges;
            adjacency[i, 3] = (i + NumEdges - 2) % NumEdges;
        }

        return adjacency;
    }
}
