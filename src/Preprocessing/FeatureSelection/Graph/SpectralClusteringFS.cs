using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Graph;

/// <summary>
/// Spectral Clustering-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses spectral graph theory to cluster features and select representative
/// features from each cluster based on eigenvector analysis.
/// </para>
/// <para><b>For Beginners:</b> Spectral clustering looks at features as points
/// in a graph and uses special math (eigenvectors) to find natural groups.
/// We then pick the most representative feature from each group.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpectralClusteringFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly int _nClusters;
    private readonly double _sigma;

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

    public SpectralClusteringFS(
        int nClusters = 10,
        double sigma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nClusters < 1)
            throw new ArgumentException("Number of clusters must be at least 1.", nameof(nClusters));

        _nClusters = nClusters;
        _sigma = sigma;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // Similarity between features: exp(-||x_j1 - x_j2||^2 / (2 sigma^2 n))
        var distances = FeatureSelectionTensorOps.SquaredDistances(Engine.TensorTranspose(x));
        var w = Engine.TensorExp(Engine.TensorMultiplyScalar(distances, NumOps.FromDouble(-1.0 / (2 * _sigma * _sigma * n))));

        // Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2), the root offset by 1e-10
        var degree = Engine.ReduceSum(w, new[] { 1 }, false);
        var scale = Engine.TensorAddScalar(Engine.TensorSqrt(Engine.TensorOuter(degree, degree)), NumOps.FromDouble(1e-10));
        var laplacian = Engine.TensorSubtract(Engine.TensorEye<T>(p), Engine.TensorDivide(w, scale));

        // Bottom k eigenvectors (eigenvalues ascending)
        int k = Math.Min(_nClusters, p);
        var (_, vectors) = LinalgDispatch<T>.Eigh(laplacian);
        var embedding = Engine.TensorSlice(vectors, new[] { 0, 0 }, new[] { p, k });

        // K-means on eigenvector embedding
        _clusterAssignments = KMeansClustering(embedding, p, k);

        // Feature scores: sample variance
        var variance = Engine.TensorDivideScalar(FilterStatisticTensorOps.ColumnSumOfSquaredDeviations(x), NumOps.FromDouble(n - 1));
        _featureScores = variance.ToVector();

        // Select the highest-variance feature from each cluster (ties to the lower index)
        T excluded = NumOps.FromDouble(double.NegativeInfinity);
        var members = FilterStatisticTensorOps.ClassOneHot<T>(_clusterAssignments, k);
        var selectedList = new List<int>();
        for (int c = 0; c < k; c++)
        {
            var inCluster = PreprocessingTensorOps.ColumnVector(members, c);
            if (FilterStatisticTensorOps.MaskPositions(inCluster).Length > 0)
                selectedList.Add(FilterStatisticTensorOps.FirstArgMax(FeatureSelectionTensorOps.KeepOrFill(variance, inCluster, excluded)));
        }

        _selectedIndices = selectedList.OrderBy(j => j).ToArray();
        IsFitted = true;
    }

    // Lloyd k-means on the rows of a [p, k] embedding: k distinct random rows start as centroids, then 50 rounds of
    // nearest-centroid assignment (ties to the lower cluster) and centroid means (an empty cluster keeps its centroid)
    private int[] KMeansClustering(Tensor<T> data, int p, int k)
    {
        var rand = RandomHelper.CreateSecureRandom();
        var initIdx = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(k).ToArray();
        var centroids = FeatureSelectionTensorOps.Rows(data, initIdx);
        var assignments = new int[p];
        T zero = NumOps.Zero;
        for (int iter = 0; iter < 50; iter++)
        {
            // Assign
            var distances = FeatureSelectionTensorOps.SquaredDistancesBetween(data, centroids);
            assignments = FeatureSelectionTensorOps.FirstArgMinRows(distances).Index;

            // Update
            var members = FilterStatisticTensorOps.ClassOneHot<T>(assignments, k);
            var counts = FeatureSelectionTensorOps.Counts(members, 0);
            var sums = Engine.TensorMatMul(Engine.TensorTranspose(members), data);
            var means = FilterStatisticTensorOps.GuardedDivide(sums, FilterStatisticTensorOps.ColumnBroadcast(counts, k), 0.5, zero);
            centroids = Engine.TensorWhere(FilterStatisticTensorOps.ColumnBroadcast(Engine.TensorGreaterThan(counts, zero), k), means, centroids);
        }

        return assignments;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralClusteringFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpectralClusteringFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralClusteringFS has not been fitted.");

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
