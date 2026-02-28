using AiDotNet.CausalDiscovery;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for CausalGraph, CausalDiscoveryResult, CausalDiscoverySelector,
/// InterventionalDistribution, and CausalDiscoveryAlgorithmFactory.
/// Tests verify graph operations, topological sort, interventional queries, and factory correctness.
/// </summary>
public class CausalGraphTests
{
    private static Matrix<double> CreateSmallAdjacency()
    {
        // 3x3 adjacency: X0 -> X1, X1 -> X2
        return new Matrix<double>(new double[,]
        {
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
            { 0.0, 0.0, 0.0 },
        });
    }

    private static Matrix<double> CreateSyntheticData()
    {
        int n = 30;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = 2.0 * data[i, 0] + 1.0;
            data[i, 2] = data[i, 0] + 0.3 * data[i, 1] + 0.5;
        }

        return new Matrix<double>(data);
    }

    #region CausalGraph Structure Tests

    [Fact]
    public void CausalGraph_Construction_PreservesAdjacencyStructure()
    {
        var adj = CreateSmallAdjacency();
        var names = new[] { "X0", "X1", "X2" };
        var graph = new CausalGraph<double>(adj, names);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.NumVariables);

        // Verify edges: X0->X1, X1->X2
        Assert.True(graph.HasEdge(0, 1), "Expected edge X0->X1");
        Assert.True(graph.HasEdge(1, 2), "Expected edge X1->X2");
        Assert.False(graph.HasEdge(0, 2), "Should not have direct edge X0->X2");
        Assert.False(graph.HasEdge(2, 0), "Should not have reverse edge X2->X0");
        Assert.False(graph.HasEdge(1, 0), "Should not have reverse edge X1->X0");
        Assert.False(graph.HasEdge(2, 1), "Should not have reverse edge X2->X1");
    }

    [Fact]
    public void CausalGraph_GetParents_ReturnsCorrectParents()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        // X0 has no parents (root cause)
        Assert.Empty(graph.GetParents(0));

        // X1's only parent is X0
        var x1Parents = graph.GetParents(1);
        Assert.Single(x1Parents);
        Assert.Equal(0, x1Parents[0]);

        // X2's only parent is X1
        var x2Parents = graph.GetParents(2);
        Assert.Single(x2Parents);
        Assert.Equal(1, x2Parents[0]);

        // String-based API
        Assert.Empty(graph.GetParents("X0"));
        Assert.Equal(["X0"], graph.GetParents("X1"));
        Assert.Equal(["X1"], graph.GetParents("X2"));
    }

    [Fact]
    public void CausalGraph_GetChildren_ReturnsCorrectChildren()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        // X0's only child is X1
        var x0Children = graph.GetChildren(0);
        Assert.Single(x0Children);
        Assert.Equal(1, x0Children[0]);

        // X1's only child is X2
        var x1Children = graph.GetChildren(1);
        Assert.Single(x1Children);
        Assert.Equal(2, x1Children[0]);

        // X2 has no children (leaf)
        Assert.Empty(graph.GetChildren(2));
    }

    [Fact]
    public void CausalGraph_GetAncestors_ReturnsTransitiveCauses()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        // X0 has no ancestors
        Assert.Empty(graph.GetAncestors(0));

        // X1's ancestor is X0
        Assert.Contains(0, graph.GetAncestors(1));

        // X2's ancestors are X0 and X1 (transitive)
        var x2Ancestors = graph.GetAncestors(2);
        Assert.Equal(2, x2Ancestors.Length);
        Assert.Contains(0, x2Ancestors);
        Assert.Contains(1, x2Ancestors);
    }

    [Fact]
    public void CausalGraph_GetDescendants_ReturnsTransitiveEffects()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        // X0's descendants are X1 and X2 (transitive)
        var x0Descendants = graph.GetDescendants(0);
        Assert.Equal(2, x0Descendants.Length);
        Assert.Contains(1, x0Descendants);
        Assert.Contains(2, x0Descendants);

        // X1's only descendant is X2
        var x1Descendants = graph.GetDescendants(1);
        Assert.Single(x1Descendants);
        Assert.Equal(2, x1Descendants[0]);

        // X2 has no descendants
        Assert.Empty(graph.GetDescendants(2));
    }

    [Fact]
    public void CausalGraph_GetMarkovBlanket_IncludesParentsAndChildren()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        // X1's Markov blanket: parent X0, child X2
        var x1Blanket = graph.GetMarkovBlanket(1);
        Assert.Contains(0, x1Blanket); // parent
        Assert.Contains(2, x1Blanket); // child
    }

    [Fact]
    public void CausalGraph_EdgeCount_And_Density()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        Assert.Equal(2, graph.EdgeCount); // X0->X1, X1->X2
        Assert.Equal(2.0 / 6.0, graph.Density, 1e-10); // 2 edges out of 3*2=6 possible
    }

    [Fact]
    public void CausalGraph_GetEdges_ReturnsAllEdgesWithWeights()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        var edges = graph.GetEdges();
        Assert.Equal(2, edges.Count);

        // Verify edge (0,1) with weight 1.0
        Assert.Contains(edges, e => e.From == 0 && e.To == 1 && Math.Abs(e.Weight - 1.0) < 1e-10);
        // Verify edge (1,2) with weight 1.0
        Assert.Contains(edges, e => e.From == 1 && e.To == 2 && Math.Abs(e.Weight - 1.0) < 1e-10);

        // Named edges
        var named = graph.GetNamedEdges();
        Assert.Equal(2, named.Count);
        Assert.Contains(named, e => e.From == "X0" && e.To == "X1");
        Assert.Contains(named, e => e.From == "X1" && e.To == "X2");
    }

    [Fact]
    public void CausalGraph_GetNodeImportance_X0HasHighestImportance()
    {
        // X0 -> X1 (weight 1.0), X1 -> X2 (weight 1.0)
        // X0 out-degree weight = 1.0, X1 out-degree weight = 1.0, X2 out-degree weight = 0
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        var importance = graph.GetNodeImportance();
        Assert.True(importance[0] > 0, "X0 should have positive importance (it causes X1)");
        Assert.True(importance[1] > 0, "X1 should have positive importance (it causes X2)");
        Assert.Equal(0.0, importance[2]); // X2 causes nothing
    }

    #endregion

    #region TopologicalSort Tests

    [Fact]
    public void CausalGraph_TopologicalSort_CausesBeforeEffects()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);

        var order = graph.TopologicalSort();
        Assert.Equal(3, order.Length);

        // X0 must come before X1 (X0 causes X1)
        int x0Pos = Array.IndexOf(order, 0);
        int x1Pos = Array.IndexOf(order, 1);
        int x2Pos = Array.IndexOf(order, 2);

        Assert.True(x0Pos < x1Pos, "X0 should precede X1 in topological order");
        Assert.True(x1Pos < x2Pos, "X1 should precede X2 in topological order");
    }

    [Fact]
    public void CausalGraph_IsDAG_TrueForAcyclicGraph()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);
        Assert.True(graph.IsDAG(), "Chain graph X0->X1->X2 should be a DAG");
    }

    [Fact]
    public void CausalGraph_IsDAG_FalseForCyclicGraph()
    {
        // Create cycle: X0->X1, X1->X2, X2->X0
        var adj = new Matrix<double>(new double[,]
        {
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
            { 1.0, 0.0, 0.0 },
        });
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);
        Assert.False(graph.IsDAG(), "Cyclic graph should not be a DAG");
    }

    #endregion

    #region ComputeInterventionalDistribution Tests

    [Fact]
    public void CausalGraph_InterventionalDistribution_ProducesValidSamples()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);
        var data = CreateSyntheticData();

        // Intervene: do(X0 = 2.0), observe effect on X2
        var dist = graph.ComputeInterventionalDistribution(0, 2.0, 2, data);

        Assert.NotNull(dist);
        Assert.Equal("X0", dist.InterventionVariableName);
        Assert.Equal("X2", dist.TargetVariableName);
        Assert.Equal(data.Rows, dist.Samples.Length);

        // Samples should not contain NaN or Infinity
        foreach (var sample in dist.Samples)
        {
            Assert.False(double.IsNaN(sample), "Interventional sample should not be NaN");
            Assert.False(double.IsInfinity(sample), "Interventional sample should not be Infinity");
        }
    }

    [Fact]
    public void CausalGraph_InterventionalDistribution_NameBasedAPI()
    {
        var adj = CreateSmallAdjacency();
        var graph = new CausalGraph<double>(adj, ["X0", "X1", "X2"]);
        var data = CreateSyntheticData();

        var dist = graph.ComputeInterventionalDistribution("X0", 2.0, "X2", data);
        Assert.NotNull(dist);
        Assert.Equal(0, dist.InterventionVariableIndex);
        Assert.Equal(2, dist.TargetVariableIndex);
    }

    #endregion

    #region CausalDiscoverySelector Tests

    [Fact]
    public void CausalDiscoverySelector_FitTransform_SelectsCausalFeatures()
    {
        var selector = new CausalDiscoverySelector<double>();
        Assert.NotNull(selector);

        var data = CreateSyntheticData();
        var target = new Vector<double>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
            target[i] = data[i, 2]; // Use X2 as target

        selector.Fit(data, target);
        Assert.NotNull(selector.SelectedIndices);
        Assert.True(selector.SelectedIndices.Length > 0, "Should select at least one causal feature");
    }

    #endregion

    #region CausalDiscoveryAlgorithmFactory Tests

    [Fact]
    public void Factory_CreatePC_ReturnsCorrectType()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotNull(algorithm);
        var algorithm2 = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotSame(algorithm, algorithm2);
    }

    [Fact]
    public void Factory_CreateGES_ReturnsCorrectType()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.GES);
        Assert.NotNull(algorithm);
        var pcAlgorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotEqual(algorithm.GetType(), pcAlgorithm.GetType());
    }

    [Fact]
    public void Factory_Discover_FoundEdgesAndGraphAPIWorks()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        var data = CreateSyntheticData();
        var names = new[] { "X0", "X1", "X2" };
        var graph = algorithm.DiscoverStructure(data, names);

        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void CausalGraph_NonSquareMatrix_Throws()
    {
        var adj = new Matrix<double>(new double[,] { { 0, 1 }, { 0, 0 }, { 0, 0 } });
        Assert.Throws<ArgumentException>(() => new CausalGraph<double>(adj, ["A", "B", "C"]));
    }

    [Fact]
    public void CausalGraph_MismatchedNames_Throws()
    {
        var adj = CreateSmallAdjacency();
        Assert.Throws<ArgumentException>(() => new CausalGraph<double>(adj, ["A", "B"]));
    }

    [Fact]
    public void CausalGraph_DuplicateNames_Throws()
    {
        var adj = CreateSmallAdjacency();
        Assert.Throws<ArgumentException>(() => new CausalGraph<double>(adj, ["A", "A", "B"]));
    }

    #endregion
}
