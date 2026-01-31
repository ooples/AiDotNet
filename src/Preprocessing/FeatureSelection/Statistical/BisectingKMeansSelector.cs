using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class BisectingKMeansSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Transpose: treat features as points (p points in n-dimensional space)
        var featureVectors = new double[p][];
        for (int j = 0; j < p; j++)
        {
            featureVectors[j] = new double[n];
            for (int i = 0; i < n; i++)
                featureVectors[j][i] = X[i, j];
        }

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
            var (c1, c2) = BisectCluster(featureVectors, clusters[largestIdx], n, rand);

            clusters.RemoveAt(largestIdx);
            if (c1.Count > 0) clusters.Add(c1);
            if (c2.Count > 0) clusters.Add(c2);
        }

        // Score features based on variance within data
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, j];
            mean /= n;
            for (int i = 0; i < n; i++) _featureScores[j] += (X[i, j] - mean) * (X[i, j] - mean);
            _featureScores[j] /= (n - 1);
        }

        // Select best feature from each cluster
        var selected = new List<int>();
        foreach (var cluster in clusters)
        {
            if (cluster.Count == 0) continue;
            int bestFeature = cluster.OrderByDescending(j => _featureScores[j]).First();
            selected.Add(bestFeature);
        }

        // If we need more features, add highest scoring remaining ones
        while (selected.Count < numClusters)
        {
            var remaining = Enumerable.Range(0, p)
                .Where(j => !selected.Contains(j))
                .OrderByDescending(j => _featureScores[j])
                .ToArray();
            if (remaining.Length == 0) break;
            selected.Add(remaining[0]);
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private (List<int>, List<int>) BisectCluster(double[][] featureVectors, List<int> cluster, int dim, Random rand)
    {
        if (cluster.Count < 2)
            return (cluster, new List<int>());

        // Initialize two centroids randomly
        int idx1 = cluster[rand.Next(cluster.Count)];
        int idx2 = cluster[rand.Next(cluster.Count)];
        while (idx2 == idx1 && cluster.Count > 1)
            idx2 = cluster[rand.Next(cluster.Count)];

        var centroid1 = (double[])featureVectors[idx1].Clone();
        var centroid2 = (double[])featureVectors[idx2].Clone();

        var c1 = new List<int>();
        var c2 = new List<int>();

        for (int iter = 0; iter < _nIterations; iter++)
        {
            c1.Clear();
            c2.Clear();

            // Assign
            foreach (int j in cluster)
            {
                double d1 = ComputeDistance(featureVectors[j], centroid1, dim);
                double d2 = ComputeDistance(featureVectors[j], centroid2, dim);
                if (d1 <= d2)
                    c1.Add(j);
                else
                    c2.Add(j);
            }

            // Handle empty clusters
            if (c1.Count == 0 || c2.Count == 0)
            {
                c1 = cluster.Take(cluster.Count / 2).ToList();
                c2 = cluster.Skip(cluster.Count / 2).ToList();
                break;
            }

            // Update centroids
            UpdateCentroid(featureVectors, c1, centroid1, dim);
            UpdateCentroid(featureVectors, c2, centroid2, dim);
        }

        return (c1, c2);
    }

    private double ComputeDistance(double[] a, double[] b, int dim)
    {
        double dist = 0;
        for (int i = 0; i < dim; i++)
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.Sqrt(dist);
    }

    private void UpdateCentroid(double[][] vectors, List<int> cluster, double[] centroid, int dim)
    {
        Array.Clear(centroid, 0, dim);
        foreach (int j in cluster)
            for (int i = 0; i < dim; i++)
                centroid[i] += vectors[j][i];
        for (int i = 0; i < dim; i++)
            centroid[i] /= cluster.Count;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BisectingKMeansSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BisectingKMeansSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BisectingKMeansSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
