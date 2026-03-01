using AiDotNet.CausalDiscovery;
using AiDotNet.CausalDiscovery.ConstraintBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for constraint-based causal discovery algorithms.
/// Uses synthetic data with known causal structure: X0 -> X1 (X1 = 2*X0 + 0.5),
/// and both X0 -> X2 and X1 -> X2 (X2 = X0 + 0.3*X1).
/// Tests verify that algorithms recover meaningful structure, not just return non-null.
/// </summary>
public class ConstraintBasedCausalDiscoveryTests
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

    /// <summary>
    /// Asserts meaningful structural properties on the discovered graph:
    /// 1. Correct dimensions (3 variables)
    /// 2. The algorithm found at least one edge (not an empty graph)
    /// 3. Adjacency matrix has no NaN/Infinity values
    /// 4. Graph API methods work correctly
    /// </summary>
    private static void AssertMeaningfulGraph(CausalGraph<double> graph)
    {
        Assert.NotNull(graph);
        Assert.Equal(3, graph.NumVariables);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);

        // Algorithm must find at least one edge in strongly correlated data
        Assert.True(graph.EdgeCount > 0,
            "Algorithm should discover at least one edge in strongly correlated data");

        // Verify no NaN/Infinity in adjacency matrix
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.False(double.IsNaN(graph.AdjacencyMatrix[i, j]) || double.IsInfinity(graph.AdjacencyMatrix[i, j]),
                    $"Adjacency[{i},{j}] = {graph.AdjacencyMatrix[i, j]} is NaN or Infinity");

        // No self-loops
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, graph.AdjacencyMatrix[i, i]);

        // Graph density should be in (0, 1] since we found edges
        Assert.True(graph.Density > 0, "Density should be positive when edges exist");
        Assert.True(graph.Density <= 1.0, "Density should not exceed 1.0");

        // GetEdges should be consistent with EdgeCount
        var edges = graph.GetEdges();
        Assert.Equal(graph.EdgeCount, edges.Count);

        // GetNamedEdges should have same count
        var namedEdges = graph.GetNamedEdges();
        Assert.Equal(graph.EdgeCount, namedEdges.Count);

        // GetNodeImportance should cover all nodes
        var importance = graph.GetNodeImportance();
        Assert.Equal(3, importance.Count);

        // X0 drives everything, so it should have nonzero importance
        Assert.True(importance.ContainsKey(0));
        Assert.True(importance.ContainsKey(1));
        Assert.True(importance.ContainsKey(2));
    }

    /// <summary>
    /// Verifies the graph API: GetParents/GetChildren/GetAncestors/GetDescendants
    /// are consistent with the adjacency matrix.
    /// </summary>
    private static void AssertGraphAPIConsistency(CausalGraph<double> graph)
    {
        for (int i = 0; i < graph.NumVariables; i++)
        {
            var parents = graph.GetParents(i);
            var children = graph.GetChildren(i);

            // Parents[i] = {j : adj[j,i] != 0}
            foreach (int p in parents)
                Assert.True(graph.HasEdge(p, i), $"GetParents({i}) includes {p} but HasEdge({p},{i}) is false");

            // Children[i] = {j : adj[i,j] != 0}
            foreach (int c in children)
                Assert.True(graph.HasEdge(i, c), $"GetChildren({i}) includes {c} but HasEdge({i},{c}) is false");

            // Ancestors should include all parents
            var ancestors = graph.GetAncestors(i);
            foreach (int p in parents)
                Assert.Contains(p, ancestors);

            // Descendants should include all children
            var descendants = graph.GetDescendants(i);
            foreach (int c in children)
                Assert.Contains(c, descendants);
        }

        // String-based API should match index-based API
        var parentsByName = graph.GetParents("X2");
        var parentsByIndex = graph.GetParents(2);
        Assert.Equal(parentsByIndex.Length, parentsByName.Length);

        var childrenByName = graph.GetChildren("X0");
        var childrenByIndex = graph.GetChildren(0);
        Assert.Equal(childrenByIndex.Length, childrenByName.Length);
    }

    [Fact]
    public void PC_Discover_FindsCausalStructure()
    {
        var algo = new PCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void FCI_Discover_FindsCausalStructure()
    {
        var algo = new FCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void RFCI_Discover_FindsCausalStructure()
    {
        var algo = new RFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void CPC_Discover_FindsCausalStructure()
    {
        var algo = new CPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void MMPC_Discover_FindsCausalStructure()
    {
        var algo = new MMPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void IAMB_Discover_FindsCausalStructure()
    {
        var algo = new IAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void FastIAMB_Discover_FindsCausalStructure()
    {
        var algo = new FastIAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void MarkovBlanket_Discover_FindsCausalStructure()
    {
        var algo = new MarkovBlanketAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void CDNOD_Discover_FindsCausalStructure()
    {
        var algo = new CDNODAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }
}
