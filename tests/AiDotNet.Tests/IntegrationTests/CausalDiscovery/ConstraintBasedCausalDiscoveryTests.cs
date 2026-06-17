using AiDotNet.CausalDiscovery;
using AiDotNet.CausalDiscovery.ConstraintBased;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task PC_Discover_FindsCausalStructure()
    {
        var algo = new PCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task FCI_Discover_FindsCausalStructure()
    {
        var algo = new FCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task RFCI_Discover_FindsCausalStructure()
    {
        var algo = new RFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CPC_Discover_FindsCausalStructure()
    {
        var algo = new CPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task MMPC_Discover_FindsCausalStructure()
    {
        var algo = new MMPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task IAMB_Discover_FindsCausalStructure()
    {
        var algo = new IAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    /// <summary>
    /// Noisy i.i.d. structural equation model with the same causal chain
    /// (X0 → X1, X0/X1 → X2) but stochastic noise on every variable.
    /// Constraint-based discovery runs Fisher-z conditional-independence
    /// tests, which are DEGENERATE on the noiseless collinear ramp: every
    /// pairwise correlation is exactly ±1 (z = atanh(1) = ∞) and every
    /// partial correlation is an indeterminate 0/0, so test decisions are
    /// numerical accidents rather than statistics. Fast-IAMB's speculative
    /// multi-variable admission (Yaramakala &amp; Margaritis 2005) and
    /// CD-NOD's time-index conditioning (Huang et al. 2020 — a deterministic
    /// ramp is PURE nonstationarity, fully explained by the index, so an
    /// empty X-X graph is the paper-correct answer there) both need
    /// well-posed CI statistics to find structure.
    /// </summary>
    private static Matrix<double> CreateNoisySEMData()
    {
        int n = 200;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            double x1 = 2.0 * x0 + 0.5 + (rng.NextDouble() * 0.4 - 0.2);
            double x2 = x0 + 0.3 * x1 + (rng.NextDouble() * 0.4 - 0.2);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task FastIAMB_Discover_FindsCausalStructure()
    {
        var algo = new FastIAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateNoisySEMData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task MarkovBlanket_Discover_FindsCausalStructure()
    {
        var algo = new MarkovBlanketAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CDNOD_Discover_FindsCausalStructure()
    {
        // Noisy SEM fixture — see CreateNoisySEMData: on the deterministic ramp
        // CD-NOD's time-index surrogate explains ALL dependence, so the
        // paper-correct output there is an empty X-X graph.
        var algo = new CDNODAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateNoisySEMData(), FeatureNames);
        AssertMeaningfulGraph(graph);
        AssertGraphAPIConsistency(graph);
    }
}
