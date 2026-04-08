using AiDotNet.CausalDiscovery;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Deep math integration tests for CausalGraph: DAG validation, topological sort,
/// Markov blanket, ancestor/descendant computation, edge queries, graph properties.
/// </summary>
public class CausalDiscoveryDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Creates a simple chain graph: A -> B -> C
    /// </summary>
    private static CausalGraph<double> CreateChainGraph()
    {
        var adj = new Matrix<double>(3, 3);
        adj[0, 1] = 1.0; // A -> B
        adj[1, 2] = 1.0; // B -> C
        return new CausalGraph<double>(adj, new[] { "A", "B", "C" });
    }

    /// <summary>
    /// Creates a fork graph: B <- A -> C
    /// </summary>
    private static CausalGraph<double> CreateForkGraph()
    {
        var adj = new Matrix<double>(3, 3);
        adj[0, 1] = 1.0; // A -> B
        adj[0, 2] = 1.0; // A -> C
        return new CausalGraph<double>(adj, new[] { "A", "B", "C" });
    }

    /// <summary>
    /// Creates a collider graph: A -> C <- B
    /// </summary>
    private static CausalGraph<double> CreateColliderGraph()
    {
        var adj = new Matrix<double>(3, 3);
        adj[0, 2] = 1.0; // A -> C
        adj[1, 2] = 1.0; // B -> C
        return new CausalGraph<double>(adj, new[] { "A", "B", "C" });
    }

    /// <summary>
    /// Creates a diamond graph: A -> B, A -> C, B -> D, C -> D
    /// </summary>
    private static CausalGraph<double> CreateDiamondGraph()
    {
        var adj = new Matrix<double>(4, 4);
        adj[0, 1] = 1.0; // A -> B
        adj[0, 2] = 0.5; // A -> C (weaker)
        adj[1, 3] = 0.8; // B -> D
        adj[2, 3] = 1.2; // C -> D
        return new CausalGraph<double>(adj, new[] { "A", "B", "C", "D" });
    }

    // ============================
    // DAG Validation Tests
    // ============================

    [Fact]
    public void ChainGraph_IsDAG()
    {
        var graph = CreateChainGraph();
        Assert.True(graph.IsDAG());
    }

    [Fact]
    public void ForkGraph_IsDAG()
    {
        var graph = CreateForkGraph();
        Assert.True(graph.IsDAG());
    }

    [Fact]
    public void ColliderGraph_IsDAG()
    {
        var graph = CreateColliderGraph();
        Assert.True(graph.IsDAG());
    }

    [Fact]
    public void DiamondGraph_IsDAG()
    {
        var graph = CreateDiamondGraph();
        Assert.True(graph.IsDAG());
    }

    [Fact]
    public void EmptyGraph_IsDAG()
    {
        var adj = new Matrix<double>(3, 3); // all zeros
        var graph = new CausalGraph<double>(adj, new[] { "A", "B", "C" });
        Assert.True(graph.IsDAG());
    }

    // ============================
    // Parent/Child Relationship Tests
    // ============================

    [Fact]
    public void ChainGraph_Parents_BHasParentA()
    {
        var graph = CreateChainGraph();
        var parents = graph.GetParents(1); // B's parents
        Assert.Single(parents);
        Assert.Equal(0, parents[0]); // A
    }

    [Fact]
    public void ChainGraph_Parents_AHasNoParents()
    {
        var graph = CreateChainGraph();
        var parents = graph.GetParents(0); // A's parents
        Assert.Empty(parents);
    }

    [Fact]
    public void ChainGraph_Children_AHasChildB()
    {
        var graph = CreateChainGraph();
        var children = graph.GetChildren(0); // A's children
        Assert.Single(children);
        Assert.Equal(1, children[0]); // B
    }

    [Fact]
    public void ChainGraph_Children_CIsLeaf()
    {
        var graph = CreateChainGraph();
        var children = graph.GetChildren(2); // C's children
        Assert.Empty(children);
    }

    [Fact]
    public void ForkGraph_Children_AHasTwoChildren()
    {
        var graph = CreateForkGraph();
        var children = graph.GetChildren(0); // A's children
        Assert.Equal(2, children.Length);
        Assert.Contains(1, children); // B
        Assert.Contains(2, children); // C
    }

    [Fact]
    public void ColliderGraph_Parents_CHasTwoParents()
    {
        var graph = CreateColliderGraph();
        var parents = graph.GetParents(2); // C's parents
        Assert.Equal(2, parents.Length);
        Assert.Contains(0, parents); // A
        Assert.Contains(1, parents); // B
    }

    [Fact]
    public void ParentsByName_ReturnsCorrectNames()
    {
        var graph = CreateColliderGraph();
        var parents = graph.GetParents("C");
        Assert.Equal(2, parents.Length);
        Assert.Contains("A", parents);
        Assert.Contains("B", parents);
    }

    [Fact]
    public void ChildrenByName_ReturnsCorrectNames()
    {
        var graph = CreateForkGraph();
        var children = graph.GetChildren("A");
        Assert.Equal(2, children.Length);
        Assert.Contains("B", children);
        Assert.Contains("C", children);
    }

    // ============================
    // Ancestor/Descendant Tests
    // ============================

    [Fact]
    public void ChainGraph_Ancestors_CDescendsFromAAndB()
    {
        var graph = CreateChainGraph();
        var ancestors = graph.GetAncestors(2); // C's ancestors
        Assert.Equal(2, ancestors.Length);
        Assert.Contains(0, ancestors); // A
        Assert.Contains(1, ancestors); // B
    }

    [Fact]
    public void ChainGraph_Ancestors_RootHasNoAncestors()
    {
        var graph = CreateChainGraph();
        var ancestors = graph.GetAncestors(0); // A's ancestors
        Assert.Empty(ancestors);
    }

    [Fact]
    public void ChainGraph_Descendants_ADescendsToAllOthers()
    {
        var graph = CreateChainGraph();
        var descendants = graph.GetDescendants(0); // A's descendants
        Assert.Equal(2, descendants.Length);
        Assert.Contains(1, descendants); // B
        Assert.Contains(2, descendants); // C
    }

    [Fact]
    public void DiamondGraph_Ancestors_DDescendsFromAll()
    {
        var graph = CreateDiamondGraph();
        var ancestors = graph.GetAncestors(3); // D's ancestors
        Assert.Equal(3, ancestors.Length);
        Assert.Contains(0, ancestors); // A
        Assert.Contains(1, ancestors); // B
        Assert.Contains(2, ancestors); // C
    }

    [Fact]
    public void DiamondGraph_Descendants_ADescendsToAll()
    {
        var graph = CreateDiamondGraph();
        var descendants = graph.GetDescendants(0); // A's descendants
        Assert.Equal(3, descendants.Length);
        Assert.Contains(1, descendants); // B
        Assert.Contains(2, descendants); // C
        Assert.Contains(3, descendants); // D
    }

    // ============================
    // Markov Blanket Tests
    // ============================

    [Fact]
    public void ChainGraph_MarkovBlanket_MiddleNode()
    {
        // B's Markov blanket: parents={A}, children={C}, co-parents of children={}
        // MB(B) = {A, C}
        var graph = CreateChainGraph();
        var mb = graph.GetMarkovBlanket(1);
        Assert.Equal(2, mb.Length);
        Assert.Contains(0, mb); // A
        Assert.Contains(2, mb); // C
    }

    [Fact]
    public void ChainGraph_MarkovBlanket_Root()
    {
        // A's Markov blanket: parents={}, children={B}, co-parents of B={}
        // MB(A) = {B}
        var graph = CreateChainGraph();
        var mb = graph.GetMarkovBlanket(0);
        Assert.Single(mb);
        Assert.Equal(1, mb[0]); // B
    }

    [Fact]
    public void ChainGraph_MarkovBlanket_Leaf()
    {
        // C's Markov blanket: parents={B}, children={}, co-parents={}
        // MB(C) = {B}
        var graph = CreateChainGraph();
        var mb = graph.GetMarkovBlanket(2);
        Assert.Single(mb);
        Assert.Equal(1, mb[0]); // B
    }

    [Fact]
    public void ColliderGraph_MarkovBlanket_IncludesCoParents()
    {
        // A's Markov blanket: parents={}, children={C}, co-parents of C = {B}
        // MB(A) = {C, B}
        var graph = CreateColliderGraph();
        var mb = graph.GetMarkovBlanket(0);
        Assert.Equal(2, mb.Length);
        Assert.Contains(2, mb); // C (child)
        Assert.Contains(1, mb); // B (co-parent of C)
    }

    [Fact]
    public void ColliderGraph_MarkovBlanket_Collider()
    {
        // C's Markov blanket: parents={A,B}, children={}, co-parents={}
        // MB(C) = {A, B}
        var graph = CreateColliderGraph();
        var mb = graph.GetMarkovBlanket(2);
        Assert.Equal(2, mb.Length);
        Assert.Contains(0, mb); // A
        Assert.Contains(1, mb); // B
    }

    [Fact]
    public void DiamondGraph_MarkovBlanket_MiddleNode()
    {
        // B's Markov blanket: parents={A}, children={D}, co-parents of D={C}
        // MB(B) = {A, D, C}
        var graph = CreateDiamondGraph();
        var mb = graph.GetMarkovBlanket(1);
        Assert.Equal(3, mb.Length);
        Assert.Contains(0, mb); // A (parent)
        Assert.Contains(3, mb); // D (child)
        Assert.Contains(2, mb); // C (co-parent of D)
    }

    // ============================
    // Edge Weight Tests
    // ============================

    [Fact]
    public void DiamondGraph_EdgeWeights_HandComputed()
    {
        var graph = CreateDiamondGraph();

        Assert.Equal(1.0, graph.GetEdgeWeight(0, 1), Tolerance); // A -> B = 1.0
        Assert.Equal(0.5, graph.GetEdgeWeight(0, 2), Tolerance); // A -> C = 0.5
        Assert.Equal(0.8, graph.GetEdgeWeight(1, 3), Tolerance); // B -> D = 0.8
        Assert.Equal(1.2, graph.GetEdgeWeight(2, 3), Tolerance); // C -> D = 1.2
    }

    [Fact]
    public void DiamondGraph_NoEdge_WeightIsZero()
    {
        var graph = CreateDiamondGraph();
        Assert.Equal(0.0, graph.GetEdgeWeight(0, 3), Tolerance); // A -> D (no direct edge)
        Assert.Equal(0.0, graph.GetEdgeWeight(1, 2), Tolerance); // B -> C (no edge)
    }

    [Fact]
    public void EdgeWeightByName_MatchesIndexedWeight()
    {
        var graph = CreateDiamondGraph();
        Assert.Equal(
            graph.GetEdgeWeight(0, 2),
            graph.GetEdgeWeight("A", "C"),
            Tolerance);
    }

    [Fact]
    public void HasEdge_ExistingEdge_True()
    {
        var graph = CreateChainGraph();
        Assert.True(graph.HasEdge(0, 1)); // A -> B
        Assert.True(graph.HasEdge(1, 2)); // B -> C
    }

    [Fact]
    public void HasEdge_NonExistingEdge_False()
    {
        var graph = CreateChainGraph();
        Assert.False(graph.HasEdge(0, 2)); // A -> C (no direct)
        Assert.False(graph.HasEdge(1, 0)); // B -> A (reverse)
    }

    // ============================
    // Graph Properties Tests
    // ============================

    [Fact]
    public void NumVariables_MatchesMatrixDimension()
    {
        var graph = CreateDiamondGraph();
        Assert.Equal(4, graph.NumVariables);
    }

    [Fact]
    public void FeatureNames_MatchesConstructorNames()
    {
        var graph = CreateDiamondGraph();
        Assert.Equal(new[] { "A", "B", "C", "D" }, graph.FeatureNames);
    }

    [Fact]
    public void Constructor_NonSquareMatrix_Throws()
    {
        var adj = new Matrix<double>(2, 3);
        Assert.Throws<ArgumentException>(() =>
            new CausalGraph<double>(adj, new[] { "A", "B" }));
    }

    [Fact]
    public void Constructor_NameLengthMismatch_Throws()
    {
        var adj = new Matrix<double>(3, 3);
        Assert.Throws<ArgumentException>(() =>
            new CausalGraph<double>(adj, new[] { "A", "B" }));
    }

    [Fact]
    public void Constructor_DuplicateNames_Throws()
    {
        var adj = new Matrix<double>(2, 2);
        Assert.Throws<ArgumentException>(() =>
            new CausalGraph<double>(adj, new[] { "A", "A" }));
    }

    // ============================
    // Topological Properties
    // ============================

    [Fact]
    public void ParentsAndChildren_AreInverse()
    {
        // For every edge i -> j:
        // i is in parents(j) AND j is in children(i)
        var graph = CreateDiamondGraph();

        for (int i = 0; i < graph.NumVariables; i++)
        {
            foreach (int child in graph.GetChildren(i))
            {
                Assert.Contains(i, graph.GetParents(child));
            }

            foreach (int parent in graph.GetParents(i))
            {
                Assert.Contains(i, graph.GetChildren(parent));
            }
        }
    }

    [Fact]
    public void Ancestors_ContainAllParentsTransitively()
    {
        var graph = CreateDiamondGraph();

        for (int i = 0; i < graph.NumVariables; i++)
        {
            var ancestors = new HashSet<int>(graph.GetAncestors(i));
            var parents = graph.GetParents(i);

            // All parents should be in ancestors
            foreach (int parent in parents)
            {
                Assert.Contains(parent, ancestors);
            }
        }
    }

    [Fact]
    public void Descendants_ContainAllChildrenTransitively()
    {
        var graph = CreateDiamondGraph();

        for (int i = 0; i < graph.NumVariables; i++)
        {
            var descendants = new HashSet<int>(graph.GetDescendants(i));
            var children = graph.GetChildren(i);

            // All children should be in descendants
            foreach (int child in children)
            {
                Assert.Contains(child, descendants);
            }
        }
    }

    [Fact]
    public void MarkovBlanket_AlwaysContainsParentsAndChildren()
    {
        var graph = CreateDiamondGraph();

        for (int i = 0; i < graph.NumVariables; i++)
        {
            var mb = new HashSet<int>(graph.GetMarkovBlanket(i));
            var parents = graph.GetParents(i);
            var children = graph.GetChildren(i);

            foreach (int p in parents)
                Assert.Contains(p, mb);

            foreach (int c in children)
                Assert.Contains(c, mb);
        }
    }

    [Fact]
    public void MarkovBlanket_NeverContainsSelf()
    {
        var graph = CreateDiamondGraph();

        for (int i = 0; i < graph.NumVariables; i++)
        {
            var mb = graph.GetMarkovBlanket(i);
            Assert.DoesNotContain(i, mb);
        }
    }

    [Fact]
    public void SingleNode_Graph_AllEmpty()
    {
        var adj = new Matrix<double>(1, 1);
        var graph = new CausalGraph<double>(adj, new[] { "X" });

        Assert.True(graph.IsDAG());
        Assert.Empty(graph.GetParents(0));
        Assert.Empty(graph.GetChildren(0));
        Assert.Empty(graph.GetAncestors(0));
        Assert.Empty(graph.GetDescendants(0));
        Assert.Empty(graph.GetMarkovBlanket(0));
    }
}
