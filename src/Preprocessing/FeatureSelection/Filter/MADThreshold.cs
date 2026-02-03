using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Median Absolute Deviation (MAD) threshold feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features whose Median Absolute Deviation is below a threshold. MAD
/// is a robust measure of spread that is less sensitive to outliers than variance.
/// </para>
/// <para><b>For Beginners:</b> Instead of using variance (which can be skewed by
/// extreme values), MAD looks at how far typical values are from the middle value.
/// This makes it better for data with outliers. Features with low MAD are nearly
/// constant and don't provide useful information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MADThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    private double[]? _madValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public double[]? MADValues => _madValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MADThreshold(
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0)
            throw new ArgumentException("Threshold must be non-negative.", nameof(threshold));

        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _madValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            Array.Sort(values);
            double median = values[n / 2];

            var deviations = new double[n];
            for (int i = 0; i < n; i++)
                deviations[i] = Math.Abs(values[i] - median);

            Array.Sort(deviations);
            _madValues[j] = deviations[n / 2];
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => _madValues[j] > _threshold)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADThreshold has not been fitted.");

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
        throw new NotSupportedException("MADThreshold does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADThreshold has not been fitted.");

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
