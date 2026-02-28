using AiDotNet.CausalDiscovery;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Shared test assertions for causal discovery algorithms.
/// Verifies structural properties, graph API consistency, and edge correctness.
/// </summary>
internal static class CausalDiscoveryTestHelper
{
    /// <summary>
    /// Asserts meaningful structural properties on the discovered graph:
    /// 1. Correct dimensions
    /// 2. At least one edge found in strongly correlated data
    /// 3. No NaN/Infinity in adjacency matrix
    /// 4. No self-loops
    /// 5. Density in valid range
    /// 6. Edge list consistent with EdgeCount
    /// 7. Node importance covers all variables
    /// </summary>
    public static void AssertMeaningfulGraph(CausalGraph<double> graph, int expectedVariables = 3)
    {
        Assert.NotNull(graph);
        Assert.Equal(expectedVariables, graph.NumVariables);
        Assert.Equal(expectedVariables, graph.FeatureNames.Length);
        Assert.Equal(expectedVariables, graph.AdjacencyMatrix.Rows);
        Assert.Equal(expectedVariables, graph.AdjacencyMatrix.Columns);

        // Algorithm must find at least one edge in strongly correlated data
        Assert.True(graph.EdgeCount > 0,
            "Algorithm should discover at least one edge in strongly correlated data");

        // No NaN/Infinity in adjacency matrix
        for (int i = 0; i < expectedVariables; i++)
            for (int j = 0; j < expectedVariables; j++)
                Assert.False(double.IsNaN(graph.AdjacencyMatrix[i, j]) || double.IsInfinity(graph.AdjacencyMatrix[i, j]),
                    $"Adjacency[{i},{j}] = {graph.AdjacencyMatrix[i, j]} is NaN or Infinity");

        // No self-loops
        for (int i = 0; i < expectedVariables; i++)
            Assert.Equal(0.0, graph.AdjacencyMatrix[i, i]);

        // Density in valid range
        Assert.True(graph.Density > 0, "Density should be positive when edges exist");
        Assert.True(graph.Density <= 1.0, "Density should not exceed 1.0");

        // Edge list consistent with EdgeCount
        var edges = graph.GetEdges();
        Assert.Equal(graph.EdgeCount, edges.Count);

        var namedEdges = graph.GetNamedEdges();
        Assert.Equal(graph.EdgeCount, namedEdges.Count);

        // Node importance covers all variables
        var importance = graph.GetNodeImportance();
        Assert.Equal(expectedVariables, importance.Count);
        for (int i = 0; i < expectedVariables; i++)
            Assert.True(importance.ContainsKey(i), $"Node importance missing variable {i}");
    }

    /// <summary>
    /// Verifies the graph API: GetParents/GetChildren/GetAncestors/GetDescendants/GetMarkovBlanket
    /// are consistent with the adjacency matrix and with each other.
    /// </summary>
    public static void AssertGraphAPIConsistency(CausalGraph<double> graph)
    {
        for (int i = 0; i < graph.NumVariables; i++)
        {
            var parents = graph.GetParents(i);
            var children = graph.GetChildren(i);

            // Parents of i must have edges TO i
            foreach (int p in parents)
            {
                Assert.True(graph.HasEdge(p, i),
                    $"GetParents({i}) includes {p} but HasEdge({p},{i}) is false");
                Assert.True(Math.Abs(graph.AdjacencyMatrix[p, i]) > 0,
                    $"GetParents({i}) includes {p} but AdjacencyMatrix[{p},{i}] is zero");
            }

            // Children of i must have edges FROM i
            foreach (int c in children)
            {
                Assert.True(graph.HasEdge(i, c),
                    $"GetChildren({i}) includes {c} but HasEdge({i},{c}) is false");
                Assert.True(Math.Abs(graph.AdjacencyMatrix[i, c]) > 0,
                    $"GetChildren({i}) includes {c} but AdjacencyMatrix[{i},{c}] is zero");
            }

            // Ancestors should include all parents (transitivity)
            var ancestors = graph.GetAncestors(i);
            foreach (int p in parents)
                Assert.Contains(p, ancestors);

            // Descendants should include all children (transitivity)
            var descendants = graph.GetDescendants(i);
            foreach (int c in children)
                Assert.Contains(c, descendants);

            // Markov blanket should include all parents and all children
            var markovBlanket = graph.GetMarkovBlanket(i);
            foreach (int p in parents)
                Assert.Contains(p, markovBlanket);
            foreach (int c in children)
                Assert.Contains(c, markovBlanket);
        }

        // String-based API should be consistent with index-based API
        for (int i = 0; i < graph.NumVariables; i++)
        {
            string name = graph.FeatureNames[i];
            var parentsByName = graph.GetParents(name);
            var parentsByIndex = graph.GetParents(i);
            Assert.Equal(parentsByIndex.Length, parentsByName.Length);

            var childrenByName = graph.GetChildren(name);
            var childrenByIndex = graph.GetChildren(i);
            Assert.Equal(childrenByIndex.Length, childrenByName.Length);
        }

        // GetEdgeWeight should match HasEdge
        for (int i = 0; i < graph.NumVariables; i++)
        {
            for (int j = 0; j < graph.NumVariables; j++)
            {
                double weight = graph.AdjacencyMatrix[i, j];
                if (Math.Abs(weight) > 0)
                    Assert.True(graph.HasEdge(i, j), $"Non-zero weight at [{i},{j}] but HasEdge is false");
                else
                    Assert.False(graph.HasEdge(i, j), $"Zero weight at [{i},{j}] but HasEdge is true");
            }
        }
    }
}
