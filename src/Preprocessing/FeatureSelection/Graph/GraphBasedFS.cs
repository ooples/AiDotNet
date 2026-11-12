using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Graph;

/// <summary>
/// Graph-based Feature Selection using feature relationships.
/// </summary>
/// <remarks>
/// <para>
/// Constructs a graph where nodes are features and edges represent
/// relationships (like correlation). Uses graph centrality measures
/// to identify important features.
/// </para>
/// <para><b>For Beginners:</b> Imagine features as people in a social network.
/// Features that are "well connected" (correlated with many other features)
/// might be key features. We use social network analysis techniques
/// to find the most "influential" features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GraphBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _edgeThreshold;
    private readonly GraphCentralityMeasure _centralityMeasure;

    private Vector<T>? _centralityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? CentralityScores => _centralityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GraphBasedFS(
        int nFeaturesToSelect = 10,
        double edgeThreshold = 0.3,
        GraphCentralityMeasure centralityMeasure = GraphCentralityMeasure.PageRank,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _edgeThreshold = edgeThreshold;
        _centralityMeasure = centralityMeasure;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GraphBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // Compute feature-target correlations
        var targetCorrelations = FeatureSelectionTensorOps.AbsoluteCorrelation(x, FilterStatisticTensorOps.AsTensor(target));

        // Build adjacency matrix based on feature correlations
        var adjacency = BuildAdjacencyMatrix(x, n, p);

        // Compute centrality scores
        var centrality = _centralityMeasure switch
        {
            GraphCentralityMeasure.Degree => Engine.ReduceSum(Engine.TensorGreaterThan(adjacency, NumOps.Zero), new[] { 1 }, false),
            GraphCentralityMeasure.Eigenvector => ComputeEigenvectorCentrality(adjacency, p),
            GraphCentralityMeasure.Betweenness => ComputeBetweennessCentrality(adjacency, p),
            _ => ComputePageRank(adjacency, p)
        };

        // Combine centrality with target correlation
        var scores = Engine.TensorMultiply(centrality, targetCorrelations);
        _centralityScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // |r| between features (sample statistics, each standard deviation offset by 1e-10) where it reaches the edge
    // threshold, 0 elsewhere and on the diagonal
    private Tensor<T> BuildAdjacencyMatrix(Tensor<T> x, int n, int p)
    {
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        T dof = NumOps.FromDouble(n - 1);
        var std = Engine.TensorAddScalar(
            Engine.TensorSqrt(Engine.TensorDivideScalar(FilterStatisticTensorOps.ColumnSumOfSquaredDeviations(x), dof)),
            NumOps.FromDouble(1e-10));
        var covariance = Engine.TensorMatMul(Engine.TensorTranspose(centered), centered);
        var strength = Engine.TensorAbs(Engine.TensorDivide(
            covariance, Engine.TensorMultiplyScalar(Engine.TensorOuter(std, std), dof)));
        var offDiagonal = PreprocessingTensorOps.OneMinus(Engine.TensorEye<T>(p));
        var edge = Engine.TensorMultiply(
            PreprocessingTensorOps.OneMinus(Engine.TensorLessThan(strength, NumOps.FromDouble(_edgeThreshold))), offDiagonal);
        return Engine.TensorMultiply(strength, edge);
    }

    // PageRank (Page et al. 1999) on the weighted graph, 100 power steps from the uniform vector: the rank of a
    // node with no edges (dangling) is spread uniformly, as networkx.pagerank does.
    private Tensor<T> ComputePageRank(Tensor<T> adjacency, int p, double damping = 0.85, int maxIter = 100)
    {
        var outDegrees = Engine.ReduceSum(adjacency, new[] { 1 }, false);
        var transition = FilterStatisticTensorOps.GuardedDivide(
            adjacency, FilterStatisticTensorOps.ColumnBroadcast(outDegrees, p), 1e-10, NumOps.Zero);
        var dangling = PreprocessingTensorOps.OneMinus(Engine.TensorGreaterThan(outDegrees, NumOps.FromDouble(1e-10)));
        T d = NumOps.FromDouble(damping);
        T teleport = NumOps.FromDouble((1 - damping) / p);
        T spread = NumOps.FromDouble(damping / p);
        var scores = FilterStatisticTensorOps.Filled(new[] { p }, NumOps.FromDouble(1.0 / p));
        var transposed = Engine.TensorTranspose(transition);
        for (int iter = 0; iter < maxIter; iter++)
        {
            var flow = Engine.Reshape(Engine.TensorMatMul(transposed, Engine.Reshape(scores, new[] { p, 1 })), new[] { p });
            var danglingMass = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(scores, dangling), new[] { 0 }, true), spread);
            scores = Engine.TensorAdd(
                Engine.TensorAddScalar(Engine.TensorMultiplyScalar(flow, d), teleport),
                Engine.TensorBroadcastTo(danglingMass, new[] { p }));
        }

        return scores;
    }

    // Eigenvector centrality: power iteration with A + I from the ones vector, L2-normalised each step (the shift
    // keeps the iteration from oscillating on bipartite graphs, as networkx.eigenvector_centrality does)
    private Tensor<T> ComputeEigenvectorCentrality(Tensor<T> adjacency, int p, int maxIter = 100)
    {
        var shifted = Engine.TensorAdd(adjacency, Engine.TensorEye<T>(p));
        var scores = FilterStatisticTensorOps.Filled(new[] { p, 1 }, NumOps.One);
        T epsilon = NumOps.FromDouble(1e-10);
        for (int iter = 0; iter < maxIter; iter++)
        {
            var next = Engine.TensorMatMul(shifted, scores);
            var norm = Engine.TensorAddScalar(Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(next), new[] { 0 }, true)), epsilon);
            scores = Engine.TensorDivide(next, Engine.TensorBroadcastTo(norm, new[] { p, 1 }));
        }

        return Engine.Reshape(scores, new[] { p });
    }

    // Betweenness centrality (Freeman 1977) of the unweighted graph: sum over pairs s < t, both other than v, of
    // the share of shortest s-t paths through v, sigma_sv sigma_vt / sigma_st when d(s, v) + d(v, t) = d(s, t)
    // (networkx.betweenness_centrality, normalized=False). Path counts and distances come from powers of the
    // adjacency: the number of walks of length d(s, t) is the number of shortest paths.
    private Tensor<T> ComputeBetweennessCentrality(Tensor<T> adjacency, int p)
    {
        T zero = NumOps.Zero;
        T unreachable = NumOps.FromDouble(double.PositiveInfinity);
        var linked = Engine.TensorGreaterThan(adjacency, zero);
        var distance = Engine.TensorWhere(Engine.TensorEye<T>(p),
            FilterStatisticTensorOps.Filled(new[] { p, p }, zero), FilterStatisticTensorOps.Filled(new[] { p, p }, unreachable));
        var paths = Engine.TensorEye<T>(p);
        var walks = Engine.TensorEye<T>(p);
        var steps = Engine.TensorArange<T>(1, p, 1).ToArray();
        foreach (T step in steps)
        {
            walks = Engine.TensorMatMul(walks, linked);
            var reachedNow = Engine.TensorMultiply(
                Engine.TensorGreaterThan(walks, zero), Engine.TensorEquals(distance, unreachable));
            distance = Engine.TensorWhere(reachedNow, FilterStatisticTensorOps.Filled(new[] { p, p }, step), distance);
            paths = Engine.TensorWhere(reachedNow, walks, paths);
        }

        // only pairs s < t with v different from both
        T one = NumOps.One;
        var upper = Engine.TensorTriu(FilterStatisticTensorOps.Filled(new[] { p, p }, one), 1);
        var scores = new T[p];
        for (int v = 0; v < p; v++)
        {
            var toV = FilterStatisticTensorOps.ColumnBroadcast(PreprocessingTensorOps.ColumnVector(distance, v), p);
            var fromV = FilterStatisticTensorOps.RowBroadcast(PreprocessingTensorOps.ColumnVector(distance, v), p);
            var onPath = Engine.TensorMultiply(Engine.TensorEquals(Engine.TensorAdd(toV, fromV), distance), upper);
            var notEnd = Engine.TensorMultiply(
                FilterStatisticTensorOps.ColumnBroadcast(FeatureSelectionTensorOps.ScatterToFeatures(
                    FilterStatisticTensorOps.Filled(new[] { 1 }, zero), new[] { v }, p, one), p),
                FilterStatisticTensorOps.RowBroadcast(FeatureSelectionTensorOps.ScatterToFeatures(
                    FilterStatisticTensorOps.Filled(new[] { 1 }, zero), new[] { v }, p, one), p));
            var through = Engine.TensorMultiply(
                FilterStatisticTensorOps.ColumnBroadcast(PreprocessingTensorOps.ColumnVector(paths, v), p),
                FilterStatisticTensorOps.RowBroadcast(PreprocessingTensorOps.ColumnVector(paths, v), p));
            var share = FilterStatisticTensorOps.GuardedDivide(through, paths, 0.5, zero);
            scores[v] = Engine.TensorSum(Engine.TensorMultiply(share, Engine.TensorMultiply(onPath, notEnd)));
        }

        return new Tensor<T>(scores, new[] { p });
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GraphBasedFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GraphBasedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GraphBasedFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

/// <summary>Centrality measure GraphBasedFS ranks features by.</summary>
public enum GraphCentralityMeasure
{
    /// <summary>PageRank of the weighted correlation graph.</summary>
    PageRank,

    /// <summary>Number of neighbours.</summary>
    Degree,

    /// <summary>Principal eigenvector of the weighted adjacency.</summary>
    Eigenvector,

    /// <summary>Share of shortest paths between other features passing through the feature.</summary>
    Betweenness
}
