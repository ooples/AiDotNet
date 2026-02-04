using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Distance;

/// <summary>
/// Mahalanobis Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Mahalanobis distance contribution, which accounts
/// for correlations between features and class-specific covariances.
/// </para>
/// <para><b>For Beginners:</b> Mahalanobis distance is like Euclidean distance but
/// adjusted for how data is spread out. It's useful when features are correlated
/// or have different scales. Features that contribute more to class separation
/// considering the data's shape are selected.
/// </para>
/// </remarks>
public class MahalanobisDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _distanceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistanceContributions => _distanceContributions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MahalanobisDistanceSelector(
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
            "MahalanobisDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _distanceContributions = new double[p];

        // Compute class means for each feature
        var classMeans = new Dictionary<int, double[]>();
        foreach (var c in classes)
        {
            classMeans[c] = new double[p];
            foreach (var idx in classIndices[c])
                for (int j = 0; j < p; j++)
                    classMeans[c][j] += X[idx, j];
            for (int j = 0; j < p; j++)
                classMeans[c][j] /= classIndices[c].Count;
        }

        // Compute pooled variance for each feature
        var pooledVariance = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sumSq = 0;
            int totalCount = 0;
            foreach (var c in classes)
            {
                double mean = classMeans[c][j];
                foreach (var idx in classIndices[c])
                {
                    double diff = X[idx, j] - mean;
                    sumSq += diff * diff;
                }
                totalCount += classIndices[c].Count;
            }
            pooledVariance[j] = totalCount > classes.Count ? sumSq / (totalCount - classes.Count) : 1;
        }

        // Compute Mahalanobis-like contribution for each feature
        for (int j = 0; j < p; j++)
        {
            double totalContribution = 0;
            int pairCount = 0;

            for (int ci = 0; ci < classes.Count; ci++)
            {
                for (int cj = ci + 1; cj < classes.Count; cj++)
                {
                    double meanDiff = classMeans[classes[ci]][j] - classMeans[classes[cj]][j];
                    double variance = Math.Max(pooledVariance[j], 1e-10);
                    totalContribution += (meanDiff * meanDiff) / variance;
                    pairCount++;
                }
            }

            _distanceContributions[j] = pairCount > 0 ? totalContribution / pairCount : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _distanceContributions[j])
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
            throw new InvalidOperationException("MahalanobisDistanceSelector has not been fitted.");

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
        throw new NotSupportedException("MahalanobisDistanceSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MahalanobisDistanceSelector has not been fitted.");

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
