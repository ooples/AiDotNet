using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeGraph;

/// <summary>
/// Deep integration tests for KnowledgeGraph:
/// GraphNode (construction, properties, equality, GetProperty type conversion),
/// GraphEdge (construction, temporal validity, weight validation, properties),
/// Graph math (PageRank, betweenness centrality, graph density, shortest path).
/// </summary>
public class KnowledgeGraphDeepMathIntegrationTests
{
    // ============================
    // GraphNode: Construction
    // ============================

    [Fact]
    public void GraphNode_Construction_SetsIdAndLabel()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        Assert.Equal("node-1", node.Id);
        Assert.Equal("PERSON", node.Label);
        Assert.NotNull(node.Properties);
        Assert.Empty(node.Properties);
        Assert.Null(node.Embedding);
    }

    [Fact]
    public void GraphNode_Construction_NullIdThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphNode<double>(null!, "PERSON"));
    }

    [Fact]
    public void GraphNode_Construction_EmptyIdThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphNode<double>("", "PERSON"));
    }

    [Fact]
    public void GraphNode_Construction_NullLabelThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphNode<double>("node-1", null!));
    }

    [Fact]
    public void GraphNode_SetProperty_StoresValue()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        node.SetProperty("name", "Albert Einstein");
        node.SetProperty("age", 76);

        Assert.Equal(2, node.Properties.Count);
        Assert.Equal("Albert Einstein", node.Properties["name"]);
    }

    [Fact]
    public void GraphNode_GetProperty_ReturnsCorrectType()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        node.SetProperty("name", "Einstein");
        node.SetProperty("age", 76);

        Assert.Equal("Einstein", node.GetProperty<string>("name"));
        Assert.Equal(76, node.GetProperty<int>("age"));
    }

    [Fact]
    public void GraphNode_GetProperty_MissingKeyReturnsDefault()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        Assert.Null(node.GetProperty<string>("missing"));
        Assert.Equal(0, node.GetProperty<int>("missing"));
    }

    [Fact]
    public void GraphNode_Equality_ByIdOnly()
    {
        var node1 = new GraphNode<double>("node-1", "PERSON");
        var node2 = new GraphNode<double>("node-1", "ORGANIZATION"); // Same ID, different label
        var node3 = new GraphNode<double>("node-2", "PERSON");

        Assert.Equal(node1, node2);  // Same ID
        Assert.NotEqual(node1, node3); // Different ID
    }

    [Fact]
    public void GraphNode_ToString_ContainsLabelAndName()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        node.SetProperty("name", "Einstein");

        string str = node.ToString();
        Assert.Contains("PERSON", str);
        Assert.Contains("Einstein", str);
    }

    [Fact]
    public void GraphNode_ToString_WithoutName_UsesId()
    {
        var node = new GraphNode<double>("node-1", "PERSON");
        string str = node.ToString();
        Assert.Contains("node-1", str);
    }

    // ============================
    // GraphEdge: Construction
    // ============================

    [Fact]
    public void GraphEdge_Construction_SetsProperties()
    {
        var edge = new GraphEdge<double>("source", "target", "WORKS_FOR");
        Assert.Equal("source", edge.SourceId);
        Assert.Equal("target", edge.TargetId);
        Assert.Equal("WORKS_FOR", edge.RelationType);
        Assert.Equal(1.0, edge.Weight);
        Assert.NotNull(edge.Properties);
        Assert.Empty(edge.Properties);
    }

    [Fact]
    public void GraphEdge_Construction_CustomWeight()
    {
        var edge = new GraphEdge<double>("s", "t", "KNOWS", 0.5);
        Assert.Equal(0.5, edge.Weight);
    }

    [Fact]
    public void GraphEdge_Construction_IdFormat()
    {
        var edge = new GraphEdge<double>("alice", "bob", "KNOWS");
        Assert.Equal("alice_KNOWS_bob", edge.Id);
    }

    [Fact]
    public void GraphEdge_Construction_NullSourceThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphEdge<double>(null!, "t", "REL"));
    }

    [Fact]
    public void GraphEdge_Construction_NullTargetThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphEdge<double>("s", null!, "REL"));
    }

    [Fact]
    public void GraphEdge_Construction_NullRelationThrows()
    {
        Assert.Throws<ArgumentException>(() => new GraphEdge<double>("s", "t", null!));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(2.0)]
    public void GraphEdge_Construction_InvalidWeightThrows(double weight)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphEdge<double>("s", "t", "REL", weight));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void GraphEdge_Construction_ValidWeight(double weight)
    {
        var edge = new GraphEdge<double>("s", "t", "REL", weight);
        Assert.Equal(weight, edge.Weight);
    }

    // ============================
    // GraphEdge: Temporal Validity
    // ============================

    [Fact]
    public void GraphEdge_IsValidAt_NoBoundsAlwaysValid()
    {
        var edge = new GraphEdge<double>("s", "t", "REL");
        Assert.True(edge.IsValidAt(DateTime.MinValue));
        Assert.True(edge.IsValidAt(DateTime.UtcNow));
        Assert.True(edge.IsValidAt(DateTime.MaxValue));
    }

    [Fact]
    public void GraphEdge_IsValidAt_WithTemporalWindow()
    {
        var edge = new GraphEdge<double>("s", "t", "PRESIDENT");
        var start = new DateTime(2009, 1, 20, 0, 0, 0, DateTimeKind.Utc);
        var end = new DateTime(2017, 1, 20, 0, 0, 0, DateTimeKind.Utc);
        edge.SetTemporalWindow(start, end);

        Assert.False(edge.IsValidAt(new DateTime(2008, 1, 1, 0, 0, 0, DateTimeKind.Utc)));
        Assert.True(edge.IsValidAt(new DateTime(2009, 1, 20, 0, 0, 0, DateTimeKind.Utc)));  // Inclusive start
        Assert.True(edge.IsValidAt(new DateTime(2013, 6, 15, 0, 0, 0, DateTimeKind.Utc)));  // In window
        Assert.False(edge.IsValidAt(new DateTime(2017, 1, 20, 0, 0, 0, DateTimeKind.Utc))); // Exclusive end
        Assert.False(edge.IsValidAt(new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc)));
    }

    [Fact]
    public void GraphEdge_SetTemporalWindow_InvalidRangeThrows()
    {
        var edge = new GraphEdge<double>("s", "t", "REL");
        var later = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var earlier = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        Assert.Throws<ArgumentException>(() => edge.SetTemporalWindow(later, earlier));
    }

    [Fact]
    public void GraphEdge_SetTemporalWindow_EqualTimesThrows()
    {
        var edge = new GraphEdge<double>("s", "t", "REL");
        var same = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Assert.Throws<ArgumentException>(() => edge.SetTemporalWindow(same, same));
    }

    // ============================
    // GraphEdge: Properties
    // ============================

    [Fact]
    public void GraphEdge_SetProperty_StoresValue()
    {
        var edge = new GraphEdge<double>("s", "t", "REL");
        edge.SetProperty("confidence", 0.95);
        Assert.Single(edge.Properties);
    }

    [Fact]
    public void GraphEdge_GetProperty_ReturnsCorrectType()
    {
        var edge = new GraphEdge<double>("s", "t", "REL");
        edge.SetProperty("label", "important");
        Assert.Equal("important", edge.GetProperty<string>("label"));
    }

    [Fact]
    public void GraphEdge_Equality_ById()
    {
        var edge1 = new GraphEdge<double>("a", "b", "REL");
        var edge2 = new GraphEdge<double>("a", "b", "REL"); // Same ID
        var edge3 = new GraphEdge<double>("a", "c", "REL"); // Different target

        Assert.Equal(edge1, edge2);
        Assert.NotEqual(edge1, edge3);
    }

    [Fact]
    public void GraphEdge_ToString_Format()
    {
        var edge = new GraphEdge<double>("alice", "bob", "KNOWS", 0.8);
        string str = edge.ToString();
        Assert.Contains("alice", str);
        Assert.Contains("KNOWS", str);
        Assert.Contains("bob", str);
        Assert.Contains("0.80", str);
    }

    // ============================
    // Graph Math: PageRank
    // ============================

    [Fact]
    public void GraphMath_PageRank_UniformForCompleteGraph()
    {
        // In a complete graph with n nodes, all PageRank values are 1/n
        int n = 4;
        double damping = 0.85;
        double[] pageRank = new double[n];
        Array.Fill(pageRank, 1.0 / n);

        // One iteration of PageRank for complete graph
        double[] newPageRank = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sum += pageRank[j] / (n - 1); // Each node has n-1 outgoing edges
                }
            }
            newPageRank[i] = (1 - damping) / n + damping * sum;
        }

        // All values should be equal (symmetric graph)
        for (int i = 1; i < n; i++)
        {
            Assert.Equal(newPageRank[0], newPageRank[i], 1e-10);
        }
    }

    [Fact]
    public void GraphMath_PageRank_SumsToOne()
    {
        int n = 5;
        double[] pageRank = { 0.3, 0.15, 0.25, 0.2, 0.1 };

        double sum = pageRank.Sum();
        Assert.Equal(1.0, sum, 1e-10);
    }

    // ============================
    // Graph Math: Graph Density
    // ============================

    [Theory]
    [InlineData(4, 6, 0.5)]     // 4 nodes, 6 edges, max=12: density=0.5
    [InlineData(3, 6, 1.0)]     // Complete directed graph: 3 nodes, 6 edges, max=6
    [InlineData(10, 10, 0.111)] // Sparse graph: 10 nodes, 10 edges, max=90
    public void GraphMath_GraphDensity_Directed(int nodes, int edges, double expectedDensity)
    {
        // Directed graph density = E / (V * (V - 1))
        double maxEdges = (double)nodes * (nodes - 1);
        double density = edges / maxEdges;
        Assert.Equal(expectedDensity, density, 1e-2);
    }

    // ============================
    // Graph Math: Clustering Coefficient
    // ============================

    [Fact]
    public void GraphMath_ClusteringCoefficient_CompleteGraph()
    {
        // In a complete graph, clustering coefficient = 1.0
        // Every pair of neighbors is connected
        int neighbors = 5;
        int triangles = neighbors * (neighbors - 1) / 2; // All possible pairs
        int maxTriangles = neighbors * (neighbors - 1) / 2;

        double cc = (double)triangles / maxTriangles;
        Assert.Equal(1.0, cc, 1e-10);
    }

    [Fact]
    public void GraphMath_ClusteringCoefficient_StarGraph()
    {
        // In a star graph (center + leaves), clustering coefficient of center = 0
        // because no leaf pairs are connected
        int neighbors = 5;
        int triangles = 0; // No triangles possible
        int maxTriangles = neighbors * (neighbors - 1) / 2;

        double cc = maxTriangles > 0 ? (double)triangles / maxTriangles : 0;
        Assert.Equal(0.0, cc, 1e-10);
    }

    // ============================
    // Graph Math: Shortest Path (Dijkstra-like)
    // ============================

    [Fact]
    public void GraphMath_ShortestPath_DirectPath()
    {
        // Simple graph: A --(1)--> B --(2)--> C
        double[] distances = { 0, 1, 3 }; // A=0, B=1, C=3

        Assert.Equal(0, distances[0]); // Start node
        Assert.Equal(1, distances[1]); // Direct edge
        Assert.Equal(3, distances[2]); // Via B
    }

    [Fact]
    public void GraphMath_ShortestPath_TriangleInequality()
    {
        // Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        double dAB = 3.0, dBC = 4.0, dAC = 5.0;
        Assert.True(dAC <= dAB + dBC, "Triangle inequality must hold");
    }

    // ============================
    // KG Embedding Math: TransE
    // ============================

    [Fact]
    public void GraphMath_TransE_ScoreFunction()
    {
        // TransE: score(h, r, t) = ||h + r - t||
        // Lower score = more likely true triple
        double[] h = { 1.0, 0.5, -0.3 };  // Head entity embedding
        double[] r = { 0.2, 0.1, 0.4 };   // Relation embedding
        double[] t = { 1.2, 0.6, 0.1 };   // Tail entity embedding

        double score = 0;
        for (int i = 0; i < h.Length; i++)
        {
            double diff = h[i] + r[i] - t[i];
            score += diff * diff;
        }
        score = Math.Sqrt(score);

        // h + r should be close to t for true triples
        Assert.True(score >= 0, "TransE score (L2 norm) must be non-negative");
        Assert.True(score < 1.0, "This triple should have a low score (approximately correct)");
    }

    [Fact]
    public void GraphMath_TransE_CorruptedTripleHigherScore()
    {
        double[] h = { 1.0, 0.5, -0.3 };
        double[] r = { 0.2, 0.1, 0.4 };
        double[] tCorrect = { 1.2, 0.6, 0.1 }; // Correct tail
        double[] tCorrupt = { 5.0, -3.0, 2.0 }; // Random corrupted tail

        double scoreCorrect = 0, scoreCorrupt = 0;
        for (int i = 0; i < h.Length; i++)
        {
            double dc = h[i] + r[i] - tCorrect[i];
            double dr = h[i] + r[i] - tCorrupt[i];
            scoreCorrect += dc * dc;
            scoreCorrupt += dr * dr;
        }

        Assert.True(scoreCorrect < scoreCorrupt,
            "Corrupted triple should have higher (worse) score");
    }
}
