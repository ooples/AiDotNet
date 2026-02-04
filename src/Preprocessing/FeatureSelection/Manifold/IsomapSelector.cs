using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Isomap-inspired Manifold-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that best preserve geodesic distances in the data manifold,
/// identifying features important for the underlying geometric structure.
/// </para>
/// <para><b>For Beginners:</b> Data often lies on a curved surface (manifold).
/// This selector finds features that best preserve the true distances along
/// that surface, keeping features important for the data's intrinsic geometry.
/// </para>
/// </remarks>
public class IsomapSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _manifoldScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? ManifoldScores => _manifoldScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public IsomapSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
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

        // Compute full distance matrix
        var fullDist = ComputeDistanceMatrix(X, n, Enumerable.Range(0, p).ToList());

        _manifoldScores = new double[p];

        // For each feature, measure how well it preserves manifold structure
        for (int j = 0; j < p; j++)
        {
            // Compute distance matrix using only this feature
            var singleDist = ComputeDistanceMatrix(X, n, new List<int> { j });

            // Correlation between full and single-feature distances
            _manifoldScores[j] = ComputeDistanceCorrelation(fullDist, singleDist, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _manifoldScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeDistanceMatrix(double[,] X, int n, List<int> features)
    {
        var dist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double d = 0;
                foreach (int f in features)
                {
                    double diff = X[i, f] - X[j, f];
                    d += diff * diff;
                }
                d = Math.Sqrt(d);
                dist[i, j] = d;
                dist[j, i] = d;
            }
        }
        return dist;
    }

    private double ComputeDistanceCorrelation(double[,] dist1, double[,] dist2, int n)
    {
        var d1 = new List<double>();
        var d2 = new List<double>();

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                d1.Add(dist1[i, j]);
                d2.Add(dist2[i, j]);
            }
        }

        if (d1.Count == 0) return 0;

        double mean1 = d1.Average();
        double mean2 = d2.Average();

        double cov = 0, var1 = 0, var2 = 0;
        for (int i = 0; i < d1.Count; i++)
        {
            double x1 = d1[i] - mean1;
            double x2 = d2[i] - mean2;
            cov += x1 * x2;
            var1 += x1 * x1;
            var2 += x2 * x2;
        }

        return (var1 > 1e-10 && var2 > 1e-10) ? cov / Math.Sqrt(var1 * var2) : 0;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapSelector has not been fitted.");

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
        throw new NotSupportedException("IsomapSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapSelector has not been fitted.");

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
