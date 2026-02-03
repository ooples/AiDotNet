using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Distance;

/// <summary>
/// Chebyshev Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize class separation based on Chebyshev (L-infinity)
/// distance between class centroids.
/// </para>
/// <para><b>For Beginners:</b> Chebyshev distance is the maximum difference in any
/// single dimension. It's useful when you care about the worst-case difference
/// between classes. Features with large maximum separations are preferred.
/// </para>
/// </remarks>
public class ChebyshevDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _distanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistanceScores => _distanceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ChebyshevDistanceSelector(
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
            "ChebyshevDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _distanceScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute min and max for each class
            var classRanges = new Dictionary<int, (double min, double max)>();
            foreach (var c in classes)
            {
                var values = classIndices[c].Select(i => X[i, j]).ToList();
                classRanges[c] = (values.Min(), values.Max());
            }

            // Maximum separation between class ranges
            double maxSeparation = 0;
            for (int ci = 0; ci < classes.Count; ci++)
            {
                for (int cj = ci + 1; cj < classes.Count; cj++)
                {
                    var range1 = classRanges[classes[ci]];
                    var range2 = classRanges[classes[cj]];

                    // Gap between ranges
                    double gap = Math.Max(0, Math.Max(
                        range1.min - range2.max,
                        range2.min - range1.max));

                    // Or use maximum absolute difference
                    double maxDiff = Math.Max(
                        Math.Abs(range1.max - range2.min),
                        Math.Abs(range2.max - range1.min));

                    maxSeparation = Math.Max(maxSeparation, gap > 0 ? gap : maxDiff);
                }
            }

            _distanceScores[j] = maxSeparation;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _distanceScores[j])
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
            throw new InvalidOperationException("ChebyshevDistanceSelector has not been fitted.");

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
        throw new NotSupportedException("ChebyshevDistanceSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChebyshevDistanceSelector has not been fitted.");

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
