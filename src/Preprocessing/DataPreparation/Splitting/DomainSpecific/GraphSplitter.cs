using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Graph splitter for node-level or edge-level predictions on graph data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Graph data consists of nodes (entities) connected by edges (relationships).
/// Examples include social networks, molecular structures, and citation networks.
/// </para>
/// <para>
/// <b>Split Types:</b>
/// - Node split: Nodes are divided into train/test sets
/// - Edge split: Edges are divided (for link prediction tasks)
/// - Inductive: Test nodes are completely unseen during training
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Social network analysis
/// - Molecular property prediction
/// - Knowledge graph completion
/// - Recommendation systems
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GraphSplitter<T> : DataSplitterBase<T>
{
    /// <summary>
    /// Type of graph split to perform.
    /// </summary>
    public enum GraphSplitType
    {
        /// <summary>Split nodes into train/test sets.</summary>
        Node,
        /// <summary>Split edges into train/test sets.</summary>
        Edge,
        /// <summary>Test nodes are completely unseen (inductive setting).</summary>
        Inductive
    }

    private readonly double _testSize;
    private readonly GraphSplitType _splitType;
    private int[,]? _adjacencyMatrix;

    /// <summary>
    /// Creates a new graph splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="splitType">Type of graph split. Default is Node.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GraphSplitter(
        double testSize = 0.2,
        GraphSplitType splitType = GraphSplitType.Node,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
        _splitType = splitType;
    }

    /// <summary>
    /// Sets the adjacency matrix for graph-aware splitting.
    /// </summary>
    /// <param name="adjacencyMatrix">N x N matrix where [i,j] = 1 if node i connects to node j.</param>
    public GraphSplitter<T> WithAdjacencyMatrix(int[,] adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;
        return this;
    }

    /// <inheritdoc/>
    public override string Description => $"Graph split ({_splitType}, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        switch (_splitType)
        {
            case GraphSplitType.Node:
                return SplitNodes(X, y, nSamples, targetTestSize);
            case GraphSplitType.Edge:
                return SplitEdges(X, y, nSamples, targetTestSize);
            case GraphSplitType.Inductive:
                return SplitInductive(X, y, nSamples, targetTestSize);
            default:
                throw new InvalidOperationException($"Unknown split type: {_splitType}");
        }
    }

    private DataSplitResult<T> SplitNodes(Matrix<T> X, Vector<T>? y, int nSamples, int targetTestSize)
    {
        // Simple random node split
        var indices = GetShuffledIndices(nSamples);
        var trainIndices = indices.Take(nSamples - targetTestSize).ToArray();
        var testIndices = indices.Skip(nSamples - targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }

    private DataSplitResult<T> SplitEdges(Matrix<T> X, Vector<T>? y, int nSamples, int targetTestSize)
    {
        // For edge splitting, we assume each row represents an edge
        var indices = GetShuffledIndices(nSamples);
        var trainIndices = indices.Take(nSamples - targetTestSize).ToArray();
        var testIndices = indices.Skip(nSamples - targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }

    private DataSplitResult<T> SplitInductive(Matrix<T> X, Vector<T>? y, int nSamples, int targetTestSize)
    {
        // Inductive split: test nodes should have minimal connections to training nodes
        if (_adjacencyMatrix == null)
        {
            // Fall back to random split if no adjacency matrix
            return SplitNodes(X, y, nSamples, targetTestSize);
        }

        // Calculate node degrees
        var degrees = new int[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nSamples; j++)
            {
                degrees[i] += _adjacencyMatrix[i, j];
            }
        }

        // Sort by degree (low-degree nodes are easier to isolate)
        var sortedByDegree = Enumerable.Range(0, nSamples)
            .OrderBy(i => degrees[i])
            .ToArray();

        if (_shuffle)
        {
            // Add some randomness while preferring low-degree nodes for test
            for (int i = 0; i < sortedByDegree.Length / 2; i++)
            {
                if (_random.NextDouble() < 0.3)
                {
                    int j = _random.Next(sortedByDegree.Length / 2);
                    (sortedByDegree[i], sortedByDegree[j]) = (sortedByDegree[j], sortedByDegree[i]);
                }
            }
        }

        // Low-degree nodes go to test (easier to isolate)
        var testIndices = sortedByDegree.Take(targetTestSize).ToArray();
        var trainIndices = sortedByDegree.Skip(targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }
}
