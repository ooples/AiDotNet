using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Point-Biserial Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses point-biserial correlation for feature selection when the target
/// is binary. This is the correlation between a continuous variable and
/// a binary variable.
/// </para>
/// <para><b>For Beginners:</b> Point-biserial correlation measures how well
/// a continuous feature separates two groups (like positive vs negative cases).
/// Features with high point-biserial correlation have very different values
/// for the two classes, making them useful for classification.
/// </para>
/// </remarks>
public class PointBiserialSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CorrelationScores => _correlationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PointBiserialSelector(
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
            "PointBiserialSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            double yVal = NumOps.ToDouble(target[i]);
            y[i] = yVal > 0.5 ? 1 : 0; // Binarize
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Count groups
        int n1 = y.Count(yi => yi == 1);
        int n0 = n - n1;

        if (n0 == 0 || n1 == 0)
            throw new InvalidOperationException("Target must have both classes present.");

        _correlationScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Mean for each group
            double mean1 = 0, mean0 = 0;
            for (int i = 0; i < n; i++)
            {
                if (y[i] == 1)
                    mean1 += X[i, j];
                else
                    mean0 += X[i, j];
            }
            mean1 /= n1;
            mean0 /= n0;

            // Overall standard deviation
            double overall_mean = 0;
            for (int i = 0; i < n; i++)
                overall_mean += X[i, j];
            overall_mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += (X[i, j] - overall_mean) * (X[i, j] - overall_mean);
            variance /= n;
            double std = Math.Sqrt(variance) + 1e-10;

            // Point-biserial correlation
            _correlationScores[j] = Math.Abs((mean1 - mean0) / std * Math.Sqrt((double)n0 * n1 / (n * n)));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlationScores[j])
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
            throw new InvalidOperationException("PointBiserialSelector has not been fitted.");

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
        throw new NotSupportedException("PointBiserialSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PointBiserialSelector has not been fitted.");

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
