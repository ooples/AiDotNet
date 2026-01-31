using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Redundancy;

/// <summary>
/// Correlation Cluster based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features by clustering correlated features together and selecting
/// the best representative from each cluster.
/// </para>
/// <para><b>For Beginners:</b> Features that are highly correlated form natural
/// groups. Instead of keeping all similar features, this selector picks one
/// representative from each group, reducing redundancy while keeping diversity.
/// </para>
/// </remarks>
public class CorrelationClusterSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationThreshold;

    private double[]? _clusterRepresentativeScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationThreshold => _correlationThreshold;
    public double[]? ClusterRepresentativeScores => _clusterRepresentativeScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CorrelationClusterSelector(
        int nFeaturesToSelect = 10,
        double correlationThreshold = 0.8,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationThreshold = correlationThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CorrelationClusterSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Compute relevance for each feature
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];
            relevance[j] = Math.Abs(ComputeCorrelation(col, y));
        }

        // Compute correlation matrix
        var correlation = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            correlation[j1, j1] = 1;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                var col1 = new double[n];
                var col2 = new double[n];
                for (int i = 0; i < n; i++)
                {
                    col1[i] = X[i, j1];
                    col2[i] = X[i, j2];
                }
                double corr = Math.Abs(ComputeCorrelation(col1, col2));
                correlation[j1, j2] = corr;
                correlation[j2, j1] = corr;
            }
        }

        // Simple agglomerative clustering based on correlation threshold
        var clusters = new List<List<int>>();
        var assigned = new bool[p];

        // Sort features by relevance (most relevant first)
        var sortedByRelevance = Enumerable.Range(0, p)
            .OrderByDescending(j => relevance[j])
            .ToList();

        foreach (int j in sortedByRelevance)
        {
            if (assigned[j]) continue;

            // Start a new cluster with this feature
            var cluster = new List<int> { j };
            assigned[j] = true;

            // Find all highly correlated features
            foreach (int k in sortedByRelevance)
            {
                if (!assigned[k] && correlation[j, k] >= _correlationThreshold)
                {
                    cluster.Add(k);
                    assigned[k] = true;
                }
            }

            clusters.Add(cluster);
        }

        // Select best representative from each cluster (highest relevance)
        _clusterRepresentativeScores = new double[p];
        var representatives = new List<int>();

        foreach (var cluster in clusters.OrderByDescending(c => relevance[c[0]]))
        {
            int bestRep = cluster.OrderByDescending(j => relevance[j]).First();
            representatives.Add(bestRep);
            _clusterRepresentativeScores[bestRep] = relevance[bestRep];

            if (representatives.Count >= _nFeaturesToSelect)
                break;
        }

        _selectedIndices = representatives.Take(_nFeaturesToSelect).OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        double xMean = x.Average();
        double yMean = y.Average();

        double numerator = 0, xSumSq = 0, ySumSq = 0;
        for (int i = 0; i < n; i++)
        {
            numerator += (x[i] - xMean) * (y[i] - yMean);
            xSumSq += (x[i] - xMean) * (x[i] - xMean);
            ySumSq += (y[i] - yMean) * (y[i] - yMean);
        }

        double denominator = Math.Sqrt(xSumSq * ySumSq);
        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationClusterSelector has not been fitted.");

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
        throw new NotSupportedException("CorrelationClusterSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationClusterSelector has not been fitted.");

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
