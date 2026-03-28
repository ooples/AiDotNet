using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MeshCNNTests : NeuralNetworkModelTestBase
{
    // Default MeshCNN: InputFeatures=5, NumClasses=40, PoolTargets=[1800,1350,600]
    // Need enough edges to satisfy the first pool target (>1800)
    private const int NumEdges = 2000;

    protected override int[] InputShape => [NumEdges, 5];
    protected override int[] OutputShape => [40];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var mesh = new MeshCNN<double>();
        mesh.SetEdgeAdjacency(CreateTestEdgeAdjacency());
        return mesh;
    }

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
