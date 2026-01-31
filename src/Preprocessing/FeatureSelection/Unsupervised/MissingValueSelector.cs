using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Missing Value-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features that have too many missing values (zeros, NaN, or
/// below-threshold unique values), keeping only features with sufficient data.
/// </para>
/// <para><b>For Beginners:</b> Features with lots of missing data can't be
/// reliably used for predictions. This selector removes features where too
/// much data is missing, keeping only the ones with enough information.
/// </para>
/// </remarks>
public class MissingValueSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;
    private readonly double _missingValue;

    private double[]? _missingRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[]? MissingRatios => _missingRatios;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MissingValueSelector(
        double threshold = 0.5,
        double missingValue = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _missingValue = missingValue;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _missingRatios = new double[p];
        var selected = new List<int>();

        for (int j = 0; j < p; j++)
        {
            int missingCount = 0;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (double.IsNaN(val) || Math.Abs(val - _missingValue) < 1e-10)
                    missingCount++;
            }

            _missingRatios[j] = (double)missingCount / n;

            if (_missingRatios[j] <= _threshold)
                selected.Add(j);
        }

        _selectedIndices = selected.ToArray();
        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueSelector has not been fitted.");

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
        throw new NotSupportedException("MissingValueSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueSelector has not been fitted.");

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
