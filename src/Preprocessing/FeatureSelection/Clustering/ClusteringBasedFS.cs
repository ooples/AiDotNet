using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Clustering-based Feature Selection using feature grouping.
/// </summary>
/// <remarks>
/// <para>
/// Clusters similar features together and selects representative features
/// from each cluster. This reduces redundancy by keeping diverse features
/// that capture different aspects of the data.
/// </para>
/// <para><b>For Beginners:</b> Sometimes features are very similar to each other
/// (like height in inches vs height in cm). This method groups similar features
/// together and picks one from each group. This way, you get diverse features
/// without redundant information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ClusteringBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly int _nClusters;
    private readonly int _maxIterations;

    private int[]? _clusterAssignments;
    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NClusters => _nClusters;
    public int[]? ClusterAssignments => _clusterAssignments;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ClusteringBasedFS(
        int nClusters = 10,
        int maxIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nClusters < 1)
            throw new ArgumentException("Number of clusters must be at least 1.", nameof(nClusters));
        if (maxIterations < 1)
            throw new ArgumentException("Maximum iterations must be at least 1.", nameof(maxIterations));

        _nClusters = nClusters;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        if (data.Rows == 0)
            throw new ArgumentException("Data must have at least one row.", nameof(data));
        if (data.Rows < 2)
            throw new ArgumentException("ClusteringBasedFS requires at least 2 rows to compute variance/correlation.", nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Compute feature correlation matrix
        var correlations = ComputeFeatureCorrelations(X, n, p);

        // Cluster features using k-means on correlation profiles
        int nClusters = Math.Min(_nClusters, p);
        _clusterAssignments = ClusterFeatures(correlations, p, nClusters);

        // Select best feature from each cluster based on (sample) variance
        var scores = n > 1
            ? engine.TensorDivideScalar(FeatureSelectionTensorOps.ColumnSumSquaredDeviations(X), NumOps.FromDouble(n - 1))
            : new Tensor<T>(new[] { p });
        _featureScores = scores.ToVector();

        // Select one feature per cluster with highest variance
        _selectedIndices = FeatureSelectionTensorOps.BestPerGroup(scores, _clusterAssignments, nClusters)
            .OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Tensor<T> ComputeFeatureCorrelations(Tensor<T> X, int n, int p)
    {
        var engine = Engine;

        // Centered data and standard deviations (n - 1 denominator, plus 1e-10)
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }));
        T dof = NumOps.FromDouble(n - 1);
        var stds = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, false), dof)),
            PreprocessingTensorOps.Tolerance<T>(1e-10));

        // Compute correlations; the diagonal is exactly 1
        var cov = engine.TensorMatMul(engine.TensorTranspose(centered), centered);
        var denom = engine.TensorMultiplyScalar(engine.TensorOuter(stds, stds), dof);
        var corr = engine.TensorDivide(cov, denom);
        return engine.TensorWhere(engine.TensorEye<T>(p), PreprocessingTensorOps.Filled(new[] { p, p }, NumOps.One), corr);
    }

    private int[] ClusterFeatures(Tensor<T> correlations, int p, int nClusters)
    {
        var engine = Engine;
        var assignments = new int[p];

        // Initialize centroids randomly
        var rand = RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(nClusters).ToArray();
        var centroids = engine.TensorIndexSelect(correlations, new Tensor<int>(indices, new[] { nClusters }), 0);

        // K-means iterations
        T zero = NumOps.Zero;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assign features to nearest centroid (first centroid among ties)
            var best = FeatureSelectionTensorOps.FirstArgMinPerRow(
                FeatureSelectionTensorOps.SquaredDistances(correlations, centroids));
            bool changed = !best.SequenceEqual(assignments);
            assignments = best;

            if (!changed)
                break;

            // Update centroids; an empty cluster keeps its centroid
            var membership = PreprocessingTensorOps.ClassOneHot<T>(assignments, nClusters);
            var means = FeatureSelectionTensorOps.GroupMeans(membership, correlations, out var counts);
            var occupied = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorGreaterThan(counts, zero), new[] { nClusters, 1 }), new[] { nClusters, p });
            centroids = engine.TensorWhere(occupied, means, centroids);
        }

        return assignments;
    }
    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusteringBasedFS has not been fitted.");
        if (data.Columns != _nInputFeatures)
            throw new ArgumentException(
                $"Expected {_nInputFeatures} features, but got {data.Columns}.", nameof(data));

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ClusteringBasedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusteringBasedFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusteringBasedFS has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, but got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
