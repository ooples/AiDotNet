using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Silhouette Score based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize the silhouette score of clusters, indicating
/// how well clusters are separated and cohesive.
/// </para>
/// <para><b>For Beginners:</b> The silhouette score measures how similar a point is
/// to its own cluster versus other clusters. Features that give high silhouette
/// scores create well-separated, tight clusters.
/// </para>
/// </remarks>
public class SilhouetteBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _silhouetteScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SilhouetteScores => _silhouetteScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SilhouetteBasedSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SilhouetteBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();
        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        _silhouetteScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double totalSilhouette = 0;

            for (int i = 0; i < n; i++)
            {
                int ownClass = y[i];
                var ownCluster = classIndices[ownClass];

                // Compute a(i) - average distance to own cluster
                double a = 0;
                if (ownCluster.Count > 1)
                {
                    foreach (int k in ownCluster)
                    {
                        if (k != i)
                            a += Math.Abs(X[i, j] - X[k, j]);
                    }
                    a /= (ownCluster.Count - 1);
                }

                // Compute b(i) - min average distance to other clusters
                double b = double.MaxValue;
                foreach (var c in classes)
                {
                    if (c == ownClass) continue;
                    var otherCluster = classIndices[c];
                    if (otherCluster.Count == 0) continue;

                    double avgDist = otherCluster.Average(k => Math.Abs(X[i, j] - X[k, j]));
                    b = Math.Min(b, avgDist);
                }

                if (b == double.MaxValue) b = 0;

                double s = Math.Max(a, b) > 0 ? (b - a) / Math.Max(a, b) : 0;
                totalSilhouette += s;
            }

            _silhouetteScores[j] = totalSilhouette / n;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _silhouetteScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

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
        throw new NotSupportedException("SilhouetteBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

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
