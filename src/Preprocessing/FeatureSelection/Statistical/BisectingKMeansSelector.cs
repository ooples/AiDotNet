using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Bisecting K-Means Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses bisecting K-means clustering to hierarchically cluster features,
/// selecting representative features from each cluster branch.
/// </para>
/// <para><b>For Beginners:</b> Bisecting K-means repeatedly splits the largest
/// cluster into two parts. We apply this to features (treating each feature as
/// a data point) and select representatives from different branches of the
/// resulting tree, ensuring diverse feature coverage.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BisectingKMeansSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BisectingKMeansSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Treat features as points (p points in n-dimensional space)
        var featureVectors = engine.TensorTranspose(x);

        var rand = RandomHelper.CreateSecureRandom();

        // Initialize: all features in one cluster
        var clusters = new List<List<int>> { Enumerable.Range(0, p).ToList() };

        // Bisect until we have enough clusters
        int numClusters = Math.Min(_nFeaturesToSelect, p);
        while (clusters.Count < numClusters)
        {
            // Find largest cluster
            int largestIdx = 0;
            for (int c = 1; c < clusters.Count; c++)
                if (clusters[c].Count > clusters[largestIdx].Count)
                    largestIdx = c;

            if (clusters[largestIdx].Count < 2)
                break;

            // Bisect the largest cluster
            var (c1, c2) = BisectCluster(featureVectors, clusters[largestIdx], rand);

            clusters.RemoveAt(largestIdx);
            if (c1.Count > 0) clusters.Add(c1);
            if (c2.Count > 0) clusters.Add(c2);
        }

        // Score features by their sample variance
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var scores = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(centered, centered), new[] { 0 }, false), NumOps.FromDouble(n - 1));
        _featureScores = scores.ToVector();

        // Select best feature from each cluster (first of the highest-scoring members, in cluster order)
        var selected = new List<int>();
        foreach (var cluster in clusters)
        {
            if (cluster.Count == 0) continue;
            var members = cluster.ToArray();
            var memberScores = Tensor<T>.FromVector(PreprocessingTensorOps.SelectElements(_featureScores, members));
            selected.Add(members[PreprocessingTensorOps.StableOrder(memberScores, descending: true)[0]]);
        }

        // If we need more features, add highest scoring remaining ones
        if (selected.Count < numClusters)
        {
            foreach (int j in PreprocessingTensorOps.StableOrder(scores, descending: true))
            {
                if (selected.Count >= numClusters) break;
                if (!selected.Contains(j)) selected.Add(j);
            }
        }

        _selectedIndices = selected.OrderBy(index => index).ToArray();
        IsFitted = true;
    }

    private (List<int>, List<int>) BisectCluster(Tensor<T> featureVectors, List<int> cluster, Random rand)
    {
        if (cluster.Count < 2)
            return (cluster, new List<int>());

        // Initialize two centroids randomly
        int idx1 = cluster[rand.Next(cluster.Count)];
        int idx2 = cluster[rand.Next(cluster.Count)];
        while (idx2 == idx1 && cluster.Count > 1)
            idx2 = cluster[rand.Next(cluster.Count)];

        var engine = Engine;
        var members = cluster.ToArray();
        var points = engine.TensorIndexSelect(featureVectors, new Tensor<int>(members, new[] { members.Length }), 0);
        var centroid1 = PreprocessingTensorOps.Row(featureVectors, idx1);
        var centroid2 = PreprocessingTensorOps.Row(featureVectors, idx2);

        var c1 = new List<int>();
        var c2 = new List<int>();

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Assign each member to its nearer centroid (ties go to the first)
            var d1 = RowDistances(points, centroid1);
            var d2 = RowDistances(points, centroid2);
            var nearFirst = engine.TensorLessEqual(d1, d2);
            var firstPositions = PreprocessingTensorOps.MaskPositions(nearFirst);
            var secondPositions = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(nearFirst));
            c1 = firstPositions.Select(position => members[position]).ToList();
            c2 = secondPositions.Select(position => members[position]).ToList();

            // Handle empty clusters
            if (c1.Count == 0 || c2.Count == 0)
            {
                c1 = cluster.Take(cluster.Count / 2).ToList();
                c2 = cluster.Skip(cluster.Count / 2).ToList();
                break;
            }

            // Update centroids
            centroid1 = engine.ReduceMean(
                engine.TensorIndexSelect(points, new Tensor<int>(firstPositions, new[] { firstPositions.Length }), 0),
                new[] { 0 }, true);
            centroid2 = engine.ReduceMean(
                engine.TensorIndexSelect(points, new Tensor<int>(secondPositions, new[] { secondPositions.Length }), 0),
                new[] { 0 }, true);
        }

        return (c1, c2);
    }

    private Tensor<T> RowDistances(Tensor<T> points, Tensor<T> centroid)
    {
        var engine = Engine;
        var diff = engine.TensorSubtract(points, centroid);
        return engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, false));
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BisectingKMeansSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BisectingKMeansSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BisectingKMeansSelector has not been fitted.");

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
