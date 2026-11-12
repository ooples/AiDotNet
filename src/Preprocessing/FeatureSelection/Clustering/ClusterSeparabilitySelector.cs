using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Cluster Separability based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how well they separate natural clusters in the data,
/// measuring the ratio of between-cluster to within-cluster variance.
/// </para>
/// <para><b>For Beginners:</b> Good features should make clusters easy to tell apart.
/// This selector finds features where different groups of data are far from each
/// other but points within each group are close together.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ClusterSeparabilitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;

    private Vector<T>? _separabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClusters => _nClusters;
    public Vector<T>? SeparabilityScores => _separabilityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ClusterSeparabilitySelector(
        int nFeaturesToSelect = 10,
        int nClusters = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = Tensor<T>.FromMatrix(data);

        // Simple k-means clustering to find natural clusters
        var clusterLabels = SimpleKMeans(X, n, p);

        var scores = ComputeSeparability(X, clusterLabels, n, p);
        _separabilityScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private int[] SimpleKMeans(Tensor<T> X, int n, int p)
    {
        var engine = Engine;

        // Initialize centroids by spreading evenly
        var initial = new int[_nClusters];
        for (int k = 0; k < _nClusters; k++)
            initial[k] = (k * n) / _nClusters;
        var centroids = engine.TensorIndexSelect(X, new Tensor<int>(initial, new[] { _nClusters }), 0);

        // Run a few iterations of k-means
        var labels = new int[n];
        for (int iter = 0; iter < 10; iter++)
        {
            // Assign points to nearest centroid (first centroid among ties)
            labels = FeatureSelectionTensorOps.FirstArgMinPerRow(FeatureSelectionTensorOps.SquaredDistances(X, centroids));

            // Update centroids; an empty cluster's centroid becomes 0
            var membership = PreprocessingTensorOps.ClassOneHot<T>(labels, _nClusters);
            centroids = FeatureSelectionTensorOps.GroupMeans(membership, X, out _);
        }

        return labels;
    }

    private Tensor<T> ComputeSeparability(Tensor<T> X, int[] labels, int n, int p)
    {
        var engine = Engine;

        // Cluster means and sizes
        var membership = PreprocessingTensorOps.ClassOneHot<T>(labels, _nClusters);
        var clusterMeans = FeatureSelectionTensorOps.GroupMeans(membership, X, out var clusterCounts);

        // Global mean
        var globalMean = engine.ReduceMean(X, new[] { 0 }, true);

        // Between-cluster variance
        var meanDiff = engine.TensorSubtract(clusterMeans, engine.TensorBroadcastTo(globalMean, new[] { _nClusters, p }));
        var weighted = engine.TensorMultiply(engine.TensorSquare(meanDiff),
            engine.TensorBroadcastTo(engine.Reshape(clusterCounts, new[] { _nClusters, 1 }), new[] { _nClusters, p }));
        var ssb = engine.ReduceSum(weighted, new[] { 0 }, false);

        // Within-cluster variance
        var within = engine.TensorSubtract(X, engine.TensorMatMul(membership, clusterMeans));
        var ssw = engine.ReduceSum(engine.TensorSquare(within), new[] { 0 }, false);

        return engine.TensorWhere(engine.TensorGreaterThan(ssw, NumOps.FromDouble(1e-10)),
            engine.TensorDivide(ssb, ssw), new Tensor<T>(new[] { p }));
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusterSeparabilitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ClusterSeparabilitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusterSeparabilitySelector has not been fitted.");

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
