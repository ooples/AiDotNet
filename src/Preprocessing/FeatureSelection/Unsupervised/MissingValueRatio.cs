using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Missing Value Ratio filter for removing features with too many missing values.
/// </summary>
/// <remarks>
/// <para>
/// Removes features where the proportion of missing values exceeds a threshold.
/// Missing values are identified as NaN, infinity, or optionally a specified
/// sentinel value.
/// </para>
/// <para><b>For Beginners:</b> Features with many missing values are unreliable.
/// If more than X% of a feature's values are missing, it's often better to
/// remove it entirely rather than try to impute the missing data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MissingValueRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;
    private readonly double? _missingIndicator;

    private double[]? _missingRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[]? MissingRatios => _missingRatios;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MissingValueRatio(
        double threshold = 0.5,
        double? missingIndicator = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _missingIndicator = missingIndicator;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _missingRatios = new double[p];

        for (int j = 0; j < p; j++)
        {
            int missingCount = 0;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (IsMissing(val))
                    missingCount++;
            }
            _missingRatios[j] = (double)missingCount / n;
        }

        // Select features below threshold
        _selectedIndices = _missingRatios
            .Select((r, idx) => (Ratio: r, Index: idx))
            .Where(x => x.Ratio <= _threshold)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        // If nothing selected, keep lowest missing ratio
        if (_selectedIndices.Length == 0)
        {
            int minIdx = 0;
            double minRatio = _missingRatios[0];
            for (int j = 1; j < p; j++)
            {
                if (_missingRatios[j] < minRatio)
                {
                    minRatio = _missingRatios[j];
                    minIdx = j;
                }
            }
            _selectedIndices = new[] { minIdx };
        }

        IsFitted = true;
    }

    private bool IsMissing(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            return true;

        if (_missingIndicator.HasValue)
            return Math.Abs(value - _missingIndicator.Value) < 1e-10;

        return false;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueRatio has not been fitted.");

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
        throw new NotSupportedException("MissingValueRatio does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueRatio has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
