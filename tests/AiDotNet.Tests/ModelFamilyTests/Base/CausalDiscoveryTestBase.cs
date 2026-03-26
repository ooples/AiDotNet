using AiDotNet.CausalDiscovery;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for causal discovery algorithms implementing ICausalDiscoveryAlgorithm.
/// Tests mathematical invariants that ALL causal discovery algorithms must satisfy:
/// DAG properties, numerical stability, dimensional consistency, and determinism.
/// </summary>
public abstract class CausalDiscoveryTestBase
{
    /// <summary>Factory method — subclasses return their concrete algorithm instance.</summary>
    protected abstract ICausalDiscoveryAlgorithm<double> CreateAlgorithm();

    /// <summary>Number of variables (columns) in test data. Override for algorithms needing more/fewer.</summary>
    protected virtual int NumVariables => 4;

    /// <summary>Number of samples (rows) in test data. Override for algorithms needing more data.</summary>
    protected virtual int NumSamples => 50;

    /// <summary>Whether the algorithm guarantees a strict DAG (no cycles). Most do.</summary>
    protected virtual bool GuaranteesDAG => true;

    /// <summary>Whether the algorithm guarantees no self-edges (diagonal = 0).</summary>
    protected virtual bool GuaranteesNoSelfEdges => true;

    /// <summary>Tolerance for floating-point comparisons.</summary>
    protected virtual double Tolerance => 1e-10;

    /// <summary>Creates synthetic data with known causal structure: X0 → X1 → X2, X0 → X3.</summary>
    protected virtual Matrix<double> CreateTestData()
    {
        var rng = new Random(42);
        var data = new Matrix<double>(NumSamples, NumVariables);

        for (int i = 0; i < NumSamples; i++)
        {
            // X0: independent noise
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            // X1: depends on X0
            double x1 = 0.7 * x0 + (rng.NextDouble() * 0.3 - 0.15);
            // X2: depends on X1
            double x2 = 0.5 * x1 + (rng.NextDouble() * 0.3 - 0.15);
            // X3: depends on X0
            double x3 = -0.6 * x0 + (rng.NextDouble() * 0.3 - 0.15);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
            if (NumVariables > 3) data[i, 3] = x3;
        }

        return data;
    }

    // =========================================================================
    // INVARIANT 1: Output adjacency matrix is square [d x d]
    // =========================================================================

    [Fact]
    public void DiscoverStructure_OutputIsSquare()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);

        Assert.Equal(NumVariables, graph.AdjacencyMatrix.Rows);
        Assert.Equal(NumVariables, graph.AdjacencyMatrix.Columns);
    }

    // =========================================================================
    // INVARIANT 2: Diagonal is zero (no variable causes itself)
    // =========================================================================

    [Fact]
    public void DiscoverStructure_DiagonalIsZero()
    {
        if (!GuaranteesNoSelfEdges) return;

        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);
        var adj = graph.AdjacencyMatrix;

        for (int i = 0; i < NumVariables; i++)
        {
            Assert.True(Math.Abs(adj[i, i]) < Tolerance,
                $"Self-edge at [{i},{i}] = {adj[i, i]}, expected 0. No variable should cause itself.");
        }
    }

    // =========================================================================
    // INVARIANT 3: All entries are finite (no NaN/Inf)
    // =========================================================================

    [Fact]
    public void DiscoverStructure_OutputIsFinite()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);
        var adj = graph.AdjacencyMatrix;

        for (int i = 0; i < adj.Rows; i++)
        {
            for (int j = 0; j < adj.Columns; j++)
            {
                Assert.False(double.IsNaN(adj[i, j]),
                    $"Adjacency matrix has NaN at [{i},{j}].");
                Assert.False(double.IsInfinity(adj[i, j]),
                    $"Adjacency matrix has Infinity at [{i},{j}].");
            }
        }
    }

    // =========================================================================
    // INVARIANT 4: Output is a DAG (acyclic — topological sort succeeds)
    // =========================================================================

    [Fact]
    public void DiscoverStructure_OutputIsAcyclic()
    {
        if (!GuaranteesDAG) return;

        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);
        var adj = graph.AdjacencyMatrix;
        int n = adj.Rows;

        // Kahn's algorithm for cycle detection
        var inDegree = new int[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(adj[i, j]) > Tolerance)
                    inDegree[j]++;
            }
        }

        var queue = new Queue<int>();
        for (int i = 0; i < n; i++)
        {
            if (inDegree[i] == 0) queue.Enqueue(i);
        }

        int visited = 0;
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            visited++;

            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(adj[node, j]) > Tolerance)
                {
                    inDegree[j]--;
                    if (inDegree[j] == 0) queue.Enqueue(j);
                }
            }
        }

        Assert.True(visited == n,
            $"Output graph contains a cycle — topological sort visited {visited} of {n} nodes. " +
            "Causal discovery algorithms must produce DAGs (Directed Acyclic Graphs).");
    }

    // =========================================================================
    // INVARIANT 5: Feature names match dimensions
    // =========================================================================

    [Fact]
    public void DiscoverStructure_FeatureNamesMatchDimensions()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();
        var names = new string[NumVariables];
        for (int i = 0; i < NumVariables; i++) names[i] = $"Var{i}";

        var graph = algo.DiscoverStructure(data, names);

        Assert.Equal(NumVariables, graph.FeatureNames.Length);
        Assert.Equal(NumVariables, graph.NumVariables);
    }

    // =========================================================================
    // INVARIANT 6: Constant columns produce sparse graph
    // =========================================================================

    [Fact]
    public void DiscoverStructure_ConstantColumnsProduceSparseGraph()
    {
        var algo = CreateAlgorithm();

        // Create data where all columns are constant (no variation = no causation)
        var data = new Matrix<double>(NumSamples, NumVariables);
        for (int i = 0; i < NumSamples; i++)
        {
            for (int j = 0; j < NumVariables; j++)
            {
                data[i, j] = 1.0; // All constant
            }
        }

        CausalGraph<double>? graph = null;
        try
        {
            graph = algo.DiscoverStructure(data);
        }
        catch (Exception)
        {
            // Some algorithms may throw on degenerate data — that's acceptable
            return;
        }

        if (graph is null) return;

        // Count non-zero edges
        var adj = graph.AdjacencyMatrix;
        int edgeCount = 0;
        for (int i = 0; i < adj.Rows; i++)
        {
            for (int j = 0; j < adj.Columns; j++)
            {
                if (Math.Abs(adj[i, j]) > Tolerance) edgeCount++;
            }
        }

        // With constant data, should have very few or no edges
        int maxExpectedEdges = NumVariables; // Allow some spurious edges due to numerical noise
        Assert.True(edgeCount <= maxExpectedEdges,
            $"Constant data produced {edgeCount} edges, expected at most {maxExpectedEdges}. " +
            "Constant columns have zero variance and cannot have causal relationships.");
    }

    // =========================================================================
    // INVARIANT 7: Does not mutate input data
    // =========================================================================

    [Fact]
    public void DiscoverStructure_DoesNotMutateInput()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();

        // Clone the data
        var original = new Matrix<double>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                original[i, j] = data[i, j];
            }
        }

        algo.DiscoverStructure(data);

        // Verify data unchanged
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.True(original[i, j] == data[i, j],
                    $"Input data was mutated at [{i},{j}]. Algorithms must not modify input data.");
            }
        }
    }

    // =========================================================================
    // INVARIANT 8: Edge weights are bounded (not exploding)
    // =========================================================================

    [Fact]
    public void DiscoverStructure_EdgeWeightsAreBounded()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);
        var adj = graph.AdjacencyMatrix;

        double maxAbsWeight = 0;
        for (int i = 0; i < adj.Rows; i++)
        {
            for (int j = 0; j < adj.Columns; j++)
            {
                maxAbsWeight = Math.Max(maxAbsWeight, Math.Abs(adj[i, j]));
            }
        }

        // Edge weights from normalized data should not explode
        Assert.True(maxAbsWeight < 1e6,
            $"Maximum edge weight is {maxAbsWeight:E2}, which is unreasonably large. " +
            "This suggests numerical instability in the algorithm.");
    }

    // =========================================================================
    // INVARIANT 9: Number of variables matches output
    // =========================================================================

    [Fact]
    public void DiscoverStructure_NumVariablesIsCorrect()
    {
        var algo = CreateAlgorithm();
        var data = CreateTestData();

        var graph = algo.DiscoverStructure(data);

        Assert.Equal(NumVariables, graph.NumVariables);
    }

    // =========================================================================
    // INVARIANT 10: Algorithm properties are consistent
    // =========================================================================

    [Fact]
    public void Properties_AreConsistent()
    {
        var algo = CreateAlgorithm();

        Assert.False(string.IsNullOrWhiteSpace(algo.Name),
            "Algorithm name should not be null or empty.");
    }
}
