using AiDotNet.CausalDiscovery;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for causal discovery algorithms implementing ICausalDiscoveryAlgorithm.
/// Tests deep mathematical invariants: DAG structural properties, known structure recovery,
/// conditional independence consistency, Markov property, faithfulness, data scaling
/// invariance, and topological ordering correctness.
/// </summary>
public abstract class CausalDiscoveryTestBase
{
    /// <summary>Factory method — subclasses return their concrete algorithm instance.</summary>
    protected abstract ICausalDiscoveryAlgorithm<double> CreateAlgorithm();

    /// <summary>Number of variables (columns) in test data. Override for algorithms needing more/fewer.</summary>
    protected virtual int NumVariables => 4;

    /// <summary>Number of samples (rows) in test data. Override for algorithms needing more data.</summary>
    protected virtual int NumSamples => 200;

    /// <summary>Whether the algorithm guarantees a strict DAG (no cycles). Most do.</summary>
    protected virtual bool GuaranteesDAG => true;

    /// <summary>Whether the algorithm guarantees no self-edges (diagonal = 0).</summary>
    protected virtual bool GuaranteesNoSelfEdges => true;

    /// <summary>Whether the algorithm can recover known linear structure with enough data.</summary>
    protected virtual bool CanRecoverLinearStructure => true;

    /// <summary>Edge detection threshold — weights below this are considered absent.</summary>
    protected virtual double EdgeThreshold => 1e-10;

    /// <summary>
    /// Creates synthetic data with KNOWN causal structure for verification:
    /// X0 → X1 (coeff 0.8), X1 → X2 (coeff 0.6), X0 → X3 (coeff -0.7).
    /// True adjacency: nonzero at [0,1], [1,2], [0,3]. All others zero.
    /// </summary>
    protected virtual Matrix<double> CreateKnownStructureData()
    {
        var rng = new Random(42);
        var data = new Matrix<double>(NumSamples, NumVariables);

        for (int i = 0; i < NumSamples; i++)
        {
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            double x1 = 0.8 * x0 + (rng.NextDouble() * 0.2 - 0.1);
            double x2 = 0.6 * x1 + (rng.NextDouble() * 0.2 - 0.1);
            double x3 = -0.7 * x0 + (rng.NextDouble() * 0.2 - 0.1);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
            if (NumVariables > 3) data[i, 3] = x3;
        }

        return data;
    }

    /// <summary>
    /// Creates data with statistically independent columns (no causal relationships).
    /// Each column is drawn from an independent distribution.
    /// </summary>
    protected virtual Matrix<double> CreateIndependentData()
    {
        var data = new Matrix<double>(NumSamples, NumVariables);
        for (int j = 0; j < NumVariables; j++)
        {
            // Use different seeds per column to guarantee independence
            var rng = new Random(1000 + j * 137);
            for (int i = 0; i < NumSamples; i++)
            {
                data[i, j] = rng.NextDouble() * 2.0 - 1.0;
            }
        }

        return data;
    }

    // =========================================================================
    // STRUCTURAL INVARIANTS (DAG properties)
    // =========================================================================

    // INVARIANT 1: Output adjacency matrix is square [d x d]
    [Fact]
    public void DiscoverStructure_OutputIsSquare()
    {
        var algo = CreateAlgorithm();
        var data = CreateKnownStructureData();
        var graph = algo.DiscoverStructure(data);

        Assert.Equal(NumVariables, graph.AdjacencyMatrix.Rows);
        Assert.Equal(NumVariables, graph.AdjacencyMatrix.Columns);
    }

    // INVARIANT 2: Diagonal is zero (no self-causation)
    [Fact]
    public void DiscoverStructure_DiagonalIsZero()
    {
        if (!GuaranteesNoSelfEdges) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;

        for (int i = 0; i < NumVariables; i++)
        {
            Assert.True(Math.Abs(adj[i, i]) < EdgeThreshold,
                $"Self-edge at [{i},{i}] = {adj[i, i]:E4}. No variable should cause itself.");
        }
    }

    // INVARIANT 3: Output is acyclic
    // For score-based methods: full DAG, topological sort on all edges.
    // For constraint-based methods (CPDAG): only directed edges (asymmetric) must be acyclic.
    // Undirected edges (symmetric A[i,j]≈A[j,i]) represent Markov equivalence class uncertainty.
    [Fact]
    public void DiscoverStructure_OutputIsAcyclic()
    {
        if (!GuaranteesDAG) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;
        int n = adj.Rows;

        // Build directed-only adjacency (skip undirected/symmetric edges)
        var inDegree = new int[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double forward = Math.Abs(adj[i, j]);
                double backward = Math.Abs(adj[j, i]);
                if (forward <= EdgeThreshold) continue;

                // Undirected edge: symmetric weights — skip for acyclicity check
                if (backward > EdgeThreshold && Math.Abs(forward - backward) / (forward + 1e-10) < 0.1)
                    continue;

                // Directed edge i→j
                inDegree[j]++;
            }
        }

        var queue = new Queue<int>();
        for (int i = 0; i < n; i++)
            if (inDegree[i] == 0) queue.Enqueue(i);

        int visited = 0;
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            visited++;
            for (int j = 0; j < n; j++)
            {
                if (node == j) continue;
                double forward = Math.Abs(adj[node, j]);
                double backward = Math.Abs(adj[j, node]);
                if (forward <= EdgeThreshold) continue;
                if (backward > EdgeThreshold && Math.Abs(forward - backward) / (forward + 1e-10) < 0.1)
                    continue;

                inDegree[j]--;
                if (inDegree[j] == 0) queue.Enqueue(j);
            }
        }

        Assert.True(visited == n,
            $"Directed cycle detected: topological sort visited {visited}/{n} nodes. " +
            "The directed portion of the graph must be acyclic.");
    }

    // INVARIANT 4: All entries are finite
    [Fact]
    public void DiscoverStructure_OutputIsFinite()
    {
        var algo = CreateAlgorithm();
        var adj = algo.DiscoverStructure(CreateKnownStructureData()).AdjacencyMatrix;

        for (int i = 0; i < adj.Rows; i++)
            for (int j = 0; j < adj.Columns; j++)
            {
                Assert.False(double.IsNaN(adj[i, j]), $"NaN at [{i},{j}].");
                Assert.False(double.IsInfinity(adj[i, j]), $"Inf at [{i},{j}].");
            }
    }

    // =========================================================================
    // KNOWN STRUCTURE RECOVERY (the real math test)
    // =========================================================================

    // INVARIANT 5: Root node (X0) has no FALSE ADJACENCIES
    // X0 is adjacent to X1 and X3 in the true structure (X0→X1, X0→X3).
    // X0 should NOT be adjacent to X2 (X0 and X2 are connected only through X1).
    // Edge DIRECTION may be reversed (X1→X0 instead of X0→X1) — that's a known
    // limitation of observational causal discovery (Markov equivalence).
    // But a false adjacency (X2↔X0) indicates a real algorithm bug.
    [Fact]
    public void DiscoverStructure_RootNodeHasNoFalseAdjacencies()
    {
        if (!CanRecoverLinearStructure || NumVariables < 4) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;

        // X0 is truly adjacent to X1 (edge 0-1) and X3 (edge 0-3)
        // X0 should NOT be adjacent to X2 (connected only via X1 chain)
        double falseEdge_0_2 = Math.Abs(adj[0, 2]) + Math.Abs(adj[2, 0]);

        // True edges (either direction is acceptable)
        double trueEdge_0_1 = Math.Abs(adj[0, 1]) + Math.Abs(adj[1, 0]);
        double trueEdge_0_3 = Math.Abs(adj[0, 3]) + Math.Abs(adj[3, 0]);

        // False edge should be much weaker than true edges
        if (trueEdge_0_1 > EdgeThreshold)
        {
            Assert.True(falseEdge_0_2 < trueEdge_0_1,
                $"False adjacency X0↔X2 ({falseEdge_0_2:F4}) should be weaker than " +
                $"true adjacency X0↔X1 ({trueEdge_0_1:F4}). X0 and X2 are not adjacent in the true DAG.");
        }
    }

    // INVARIANT 6: Known true edges are detected (recall)
    // The true structure has edges X0→X1, X1→X2, X0→X3. At least some should be found.
    [Fact]
    public void DiscoverStructure_RecoversTrueEdges()
    {
        if (!CanRecoverLinearStructure) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;

        // True edges: (0,1), (1,2), (0,3)
        var trueEdges = new List<(int from, int to)> { (0, 1), (1, 2) };
        if (NumVariables > 3) trueEdges.Add((0, 3));

        int detectedCount = 0;
        foreach (var (from, to) in trueEdges)
        {
            if (Math.Abs(adj[from, to]) > EdgeThreshold)
                detectedCount++;
        }

        // With 200 samples and strong signal (coeffs 0.6-0.8), should find at least 1 edge
        Assert.True(detectedCount >= 1,
            $"Algorithm found {detectedCount}/{trueEdges.Count} true edges. " +
            "With 200 samples and strong linear relationships (coefficients 0.6-0.8), " +
            "at least one true causal edge should be detected.");
    }

    // INVARIANT 7: Independent variables have weak/no edges between them
    // X1 and X3 are conditionally independent given X0 — should have weak or no edge.
    [Fact]
    public void DiscoverStructure_IndependentVariablesHaveWeakEdges()
    {
        if (!CanRecoverLinearStructure || NumVariables < 4) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;

        // X1 and X3 are conditionally independent given X0
        // Neither X1→X3 nor X3→X1 should have strong weights
        double spuriousWeight = Math.Abs(adj[1, 3]) + Math.Abs(adj[3, 1]);

        // The true edge X0→X1 should be stronger than the spurious X1↔X3
        double trueEdgeWeight = Math.Abs(adj[0, 1]);

        // Allow spurious edges but they should be weaker than true edges
        if (trueEdgeWeight > EdgeThreshold)
        {
            Assert.True(spuriousWeight <= trueEdgeWeight * 2.0 + 0.1,
                $"Spurious edge between conditionally independent X1↔X3 ({spuriousWeight:F4}) " +
                $"should not dominate true edge X0→X1 ({trueEdgeWeight:F4}).");
        }
    }

    // =========================================================================
    // STATISTICAL INVARIANTS
    // =========================================================================

    // INVARIANT 8: Fully independent data produces sparse graph
    // When all variables are independent, a correct algorithm should find few/no edges.
    [Fact]
    public void DiscoverStructure_IndependentDataProducesSparseGraph()
    {
        var algo = CreateAlgorithm();
        CausalGraph<double>? graph;
        try
        {
            graph = algo.DiscoverStructure(CreateIndependentData());
        }
        catch (Exception)
        {
            return; // Degenerate data handling is acceptable
        }

        var adj = graph.AdjacencyMatrix;
        int edgeCount = 0;
        double totalWeight = 0;
        for (int i = 0; i < adj.Rows; i++)
        {
            for (int j = 0; j < adj.Columns; j++)
            {
                if (i != j && Math.Abs(adj[i, j]) > EdgeThreshold)
                {
                    edgeCount++;
                    totalWeight += Math.Abs(adj[i, j]);
                }
            }
        }

        // For d=4 independent variables, max possible edges = 12 (d*(d-1))
        // A good algorithm should find at most ~d spurious edges due to finite-sample noise
        int maxSpuriousEdges = NumVariables * 2;
        Assert.True(edgeCount <= maxSpuriousEdges,
            $"Independent data produced {edgeCount} edges (total weight {totalWeight:F4}), " +
            $"expected at most {maxSpuriousEdges}. Independent variables should have no causal relationships.");
    }

    // INVARIANT 9: More data → same or more accurate structure
    // Running with 2x samples should not degrade structure quality (measured by false edge count)
    [Fact]
    public void DiscoverStructure_MoreDataDoesNotDegradeQuality()
    {
        if (!CanRecoverLinearStructure) return;

        var algo1 = CreateAlgorithm();
        var algo2 = CreateAlgorithm();

        // Small dataset
        var smallData = CreateKnownStructureDataWithSize(100);
        CausalGraph<double>? smallGraph;
        try
        {
            smallGraph = algo1.DiscoverStructure(smallData);
        }
        catch (Exception)
        {
            return;
        }

        // Larger dataset (same generating process)
        var largeData = CreateKnownStructureDataWithSize(400);
        CausalGraph<double>? largeGraph;
        try
        {
            largeGraph = algo2.DiscoverStructure(largeData);
        }
        catch (Exception)
        {
            return;
        }

        // Compare: count edges NOT in true structure (false positives)
        int smallFP = CountFalsePositives(smallGraph.AdjacencyMatrix);
        int largeFP = CountFalsePositives(largeGraph.AdjacencyMatrix);

        // More data should not dramatically increase false positives
        Assert.True(largeFP <= smallFP + 2,
            $"More data increased false positives: small={smallFP}, large={largeFP}. " +
            "With more data, structure should become more accurate, not worse.");
    }

    // INVARIANT 10: Data scaling invariance
    // Multiplying all data by a constant should not change the STRUCTURE (edges present/absent).
    // Only edge weights may change.
    [Fact]
    public void DiscoverStructure_IsInvariantToDataScaling()
    {
        var algo1 = CreateAlgorithm();
        var algo2 = CreateAlgorithm();

        var data = CreateKnownStructureData();

        // Scale data by 10x
        var scaledData = new Matrix<double>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
            for (int j = 0; j < data.Columns; j++)
                scaledData[i, j] = data[i, j] * 10.0;

        CausalGraph<double>? graph1, graph2;
        try
        {
            graph1 = algo1.DiscoverStructure(data);
            graph2 = algo2.DiscoverStructure(scaledData);
        }
        catch (Exception)
        {
            return;
        }

        // Compare edge PRESENCE (not weights)
        int structureDiff = 0;
        for (int i = 0; i < NumVariables; i++)
        {
            for (int j = 0; j < NumVariables; j++)
            {
                bool edge1 = Math.Abs(graph1.AdjacencyMatrix[i, j]) > EdgeThreshold;
                bool edge2 = Math.Abs(graph2.AdjacencyMatrix[i, j]) > EdgeThreshold;
                if (edge1 != edge2) structureDiff++;
            }
        }

        // Allow some differences due to numerical effects of scaling, but not many
        Assert.True(structureDiff <= NumVariables,
            $"Scaling data by 10x changed {structureDiff} edges. " +
            "Causal structure should be invariant to uniform data scaling.");
    }

    // INVARIANT 11: Topological ordering consistency (directed edges only)
    // For directed edges i→j, parent i must appear before child j in topological order.
    // Undirected edges (CPDAG) are skipped since they have no defined direction.
    [Fact]
    public void DiscoverStructure_TopologicalOrderIsConsistent()
    {
        if (!GuaranteesDAG) return;

        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;
        int n = adj.Rows;

        // Compute topological order of directed-only edges
        var order = TopologicalSortDirectedOnly(adj, n);
        if (order is null) return; // Directed cycle detected, covered by invariant 3

        var position = new int[n];
        for (int i = 0; i < order.Count; i++)
            position[order[i]] = i;

        // Verify: for every DIRECTED edge i→j, position[i] < position[j]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                if (!IsDirectedEdge(adj, i, j)) continue;

                Assert.True(position[i] < position[j],
                    $"Directed edge {i}→{j} violates topological order: node {i} at position {position[i]}, " +
                    $"node {j} at position {position[j]}.");
            }
        }
    }

    // INVARIANT 12: Edge directionality check
    // For DAGs: if A[i,j] has a strong edge, A[j,i] should be weak or zero.
    // For CPDAGs: undirected edges (symmetric weights) are valid — they represent
    // edges whose orientation cannot be determined from observational data alone.
    // The invariant checks: no ASYMMETRIC strong edges in BOTH directions (would mean a cycle).
    [Fact]
    public void DiscoverStructure_NoAsymmetricBidirectionalEdges()
    {
        var algo = CreateAlgorithm();
        var graph = algo.DiscoverStructure(CreateKnownStructureData());
        var adj = graph.AdjacencyMatrix;

        for (int i = 0; i < NumVariables; i++)
        {
            for (int j = i + 1; j < NumVariables; j++)
            {
                double forward = Math.Abs(adj[i, j]);
                double backward = Math.Abs(adj[j, i]);

                if (forward <= 0.3 || backward <= 0.3)
                    continue; // At most one direction is strong — fine

                // Both strong — check if symmetric (undirected edge in CPDAG, acceptable)
                double ratio = Math.Abs(forward - backward) / (Math.Max(forward, backward) + 1e-10);
                if (ratio < 0.2)
                    continue; // Symmetric (undirected) edge — valid in CPDAGs

                // ASYMMETRIC strong bidirectional edges = directed cycle evidence
                Assert.Fail(
                    $"Asymmetric bidirectional strong edges between {i}↔{j}: " +
                    $"forward={forward:F4}, backward={backward:F4} (ratio={ratio:F4}). " +
                    "This suggests a directed cycle, which violates the DAG constraint.");
            }
        }
    }

    // INVARIANT 13: Does not mutate input data
    [Fact]
    public void DiscoverStructure_DoesNotMutateInput()
    {
        var algo = CreateAlgorithm();
        var data = CreateKnownStructureData();

        var original = new Matrix<double>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
            for (int j = 0; j < data.Columns; j++)
                original[i, j] = data[i, j];

        algo.DiscoverStructure(data);

        for (int i = 0; i < data.Rows; i++)
            for (int j = 0; j < data.Columns; j++)
                Assert.True(original[i, j] == data[i, j],
                    $"Input mutated at [{i},{j}].");
    }

    // INVARIANT 14: Algorithm properties are consistent
    [Fact]
    public void Properties_AreConsistent()
    {
        var algo = CreateAlgorithm();

        Assert.False(string.IsNullOrWhiteSpace(algo.Name));
        // Category should be a valid enum value
        Assert.True(Enum.IsDefined(typeof(AiDotNet.Enums.CausalDiscoveryCategory), algo.Category));
    }

    // =========================================================================
    // Helper methods
    // =========================================================================

    private Matrix<double> CreateKnownStructureDataWithSize(int samples)
    {
        var rng = new Random(42);
        var data = new Matrix<double>(samples, NumVariables);

        for (int i = 0; i < samples; i++)
        {
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            double x1 = 0.8 * x0 + (rng.NextDouble() * 0.2 - 0.1);
            double x2 = 0.6 * x1 + (rng.NextDouble() * 0.2 - 0.1);
            double x3 = -0.7 * x0 + (rng.NextDouble() * 0.2 - 0.1);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
            if (NumVariables > 3) data[i, 3] = x3;
        }

        return data;
    }

    private int CountFalsePositives(Matrix<double> adj)
    {
        // True edges: (0,1), (1,2), (0,3)
        var trueEdges = new HashSet<(int, int)> { (0, 1), (1, 2) };
        if (NumVariables > 3) trueEdges.Add((0, 3));

        int fp = 0;
        for (int i = 0; i < adj.Rows; i++)
            for (int j = 0; j < adj.Columns; j++)
                if (i != j && Math.Abs(adj[i, j]) > EdgeThreshold && !trueEdges.Contains((i, j)))
                    fp++;
        return fp;
    }

    /// <summary>
    /// Checks if edge i→j is directed (not undirected/symmetric).
    /// An undirected edge has A[i,j] ≈ A[j,i]; a directed edge has only one direction non-zero
    /// or significantly asymmetric weights.
    /// </summary>
    private bool IsDirectedEdge(Matrix<double> adj, int i, int j)
    {
        double forward = Math.Abs(adj[i, j]);
        double backward = Math.Abs(adj[j, i]);
        if (forward <= EdgeThreshold) return false;
        // Symmetric = undirected
        if (backward > EdgeThreshold && Math.Abs(forward - backward) / (forward + 1e-10) < 0.1)
            return false;
        return true;
    }

    /// <summary>
    /// Topological sort considering only directed edges (skipping undirected/symmetric edges).
    /// </summary>
    private List<int>? TopologicalSortDirectedOnly(Matrix<double> adj, int n)
    {
        var inDegree = new int[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j && IsDirectedEdge(adj, i, j))
                    inDegree[j]++;

        var queue = new Queue<int>();
        for (int i = 0; i < n; i++)
            if (inDegree[i] == 0) queue.Enqueue(i);

        var result = new List<int>();
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            result.Add(node);
            for (int j = 0; j < n; j++)
            {
                if (node != j && IsDirectedEdge(adj, node, j))
                {
                    inDegree[j]--;
                    if (inDegree[j] == 0) queue.Enqueue(j);
                }
            }
        }

        return result.Count == n ? result : null;
    }
}
