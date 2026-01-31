using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantile;

/// <summary>
/// Quantile Dispersion (QCD) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Quartile Coefficient of Dispersion,
/// a robust measure of relative variability: QCD = (Q3 - Q1) / (Q3 + Q1)
/// </para>
/// <para><b>For Beginners:</b> QCD measures spread relative to the central location,
/// similar to coefficient of variation but using quartiles instead of mean/std.
/// It's robust to outliers and works well when data contains extreme values.
/// Values range from 0 (no spread) to 1 (maximum relative spread).
/// </para>
/// </remarks>
public class QuantileDispersionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _qcdValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? QCDValues => _qcdValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public QuantileDispersionSelector(
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
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _qcdValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var sorted = col.OrderBy(v => v).ToArray();
            double q1 = GetQuantile(sorted, 0.25);
            double q3 = GetQuantile(sorted, 0.75);

            // QCD = (Q3 - Q1) / (Q3 + Q1)
            double sum = q3 + q1;
            if (Math.Abs(sum) < 1e-10)
            {
                // If Q1 and Q3 cancel out, use IQR directly
                _qcdValues[j] = q3 - q1;
            }
            else
            {
                _qcdValues[j] = (q3 - q1) / Math.Abs(sum);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _qcdValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double GetQuantile(double[] sortedData, double quantile)
    {
        int n = sortedData.Length;
        double index = quantile * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sortedData[lower];

        double fraction = index - lower;
        return sortedData[lower] * (1 - fraction) + sortedData[upper] * fraction;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileDispersionSelector has not been fitted.");

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
        throw new NotSupportedException("QuantileDispersionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileDispersionSelector has not been fitted.");

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
