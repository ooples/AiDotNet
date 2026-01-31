using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Boundary;

/// <summary>
/// Margin based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the margin they create between classes,
/// preferring features with larger class separation margins.
/// </para>
/// <para><b>For Beginners:</b> The margin is the distance between classes.
/// Features with larger margins make it easier to separate classes with
/// more confidence, similar to how SVM maximizes margins.
/// </para>
/// </remarks>
public class MarginSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _marginScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MarginScores => _marginScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MarginSelector(
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
            "MarginSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _marginScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Standardize
            double mean = col.Average();
            double std = Math.Sqrt(col.Select(v => (v - mean) * (v - mean)).Average());
            if (std > 1e-10)
                for (int i = 0; i < n; i++)
                    col[i] = (col[i] - mean) / std;

            // For each pair of classes, compute margin
            double totalMargin = 0;
            int pairs = 0;

            for (int ci = 0; ci < classes.Count; ci++)
            {
                for (int cj = ci + 1; cj < classes.Count; cj++)
                {
                    var class1Vals = classIndices[classes[ci]].Select(i => col[i]).ToList();
                    var class2Vals = classIndices[classes[cj]].Select(i => col[i]).ToList();

                    // Compute margin as gap between class ranges
                    double min1 = class1Vals.Min();
                    double max1 = class1Vals.Max();
                    double min2 = class2Vals.Min();
                    double max2 = class2Vals.Max();

                    // Margin = gap between closest edges
                    double margin = Math.Max(0, Math.Max(min2 - max1, min1 - max2));

                    // If overlapping, use negative margin based on overlap
                    if (margin == 0)
                    {
                        double overlap = Math.Min(max1, max2) - Math.Max(min1, min2);
                        margin = -overlap;
                    }

                    // Normalize by average spread
                    double avgSpread = ((max1 - min1) + (max2 - min2)) / 2;
                    if (avgSpread > 1e-10)
                        margin /= avgSpread;

                    totalMargin += margin;
                    pairs++;
                }
            }

            _marginScores[j] = pairs > 0 ? totalMargin / pairs : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _marginScores[j])
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
            throw new InvalidOperationException("MarginSelector has not been fitted.");

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
        throw new NotSupportedException("MarginSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MarginSelector has not been fitted.");

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
