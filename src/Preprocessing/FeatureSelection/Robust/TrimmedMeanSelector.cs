using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Trimmed Mean-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses trimmed statistics (removing extreme values) to compute robust feature
/// importance scores that are less affected by outliers.
/// </para>
/// <para><b>For Beginners:</b> When computing feature statistics, extreme values
/// (outliers) can skew results. Trimmed mean cuts off the highest and lowest
/// values before computing, giving you a more reliable picture of typical feature
/// behavior without outlier influence.
/// </para>
/// </remarks>
public class TrimmedMeanSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _trimProportion;

    private double[]? _trimmedVariances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double TrimProportion => _trimProportion;
    public double[]? TrimmedVariances => _trimmedVariances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TrimmedMeanSelector(
        int nFeaturesToSelect = 10,
        double trimProportion = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (trimProportion < 0 || trimProportion >= 0.5)
            throw new ArgumentException("Trim proportion must be between 0 and 0.5.", nameof(trimProportion));

        _nFeaturesToSelect = nFeaturesToSelect;
        _trimProportion = trimProportion;
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

        int trimCount = (int)(n * _trimProportion);
        int trimmedN = n - 2 * trimCount;

        _trimmedVariances = new double[p];

        for (int j = 0; j < p; j++)
        {
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = X[i, j];

            var sorted = values.OrderBy(v => v).Skip(trimCount).Take(trimmedN).ToArray();

            if (sorted.Length > 1)
            {
                double mean = sorted.Average();
                double variance = 0;
                foreach (double v in sorted)
                    variance += (v - mean) * (v - mean);
                _trimmedVariances[j] = variance / (sorted.Length - 1);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _trimmedVariances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrimmedMeanSelector has not been fitted.");

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
        throw new NotSupportedException("TrimmedMeanSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrimmedMeanSelector has not been fitted.");

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
