using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Kurtosis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their kurtosis values, identifying features
/// with unusual tail behavior in their distributions.
/// </para>
/// <para><b>For Beginners:</b> Kurtosis measures how heavy the tails of a distribution
/// are. High kurtosis means more extreme values (outliers), while low kurtosis means
/// fewer extremes. Features with interesting kurtosis often capture important patterns.
/// </para>
/// </remarks>
public class KurtosisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighKurtosis;

    private double[]? _kurtosisValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighKurtosis => _preferHighKurtosis;
    public double[]? KurtosisValues => _kurtosisValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KurtosisSelector(
        int nFeaturesToSelect = 10,
        bool preferHighKurtosis = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighKurtosis = preferHighKurtosis;
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

        _kurtosisValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double variance = col.Select(v => (v - mean) * (v - mean)).Average();
            double std = Math.Sqrt(variance);

            if (std < 1e-10)
            {
                _kurtosisValues[j] = 0;
                continue;
            }

            // Compute excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3
            double moment4 = col.Select(v => Math.Pow((v - mean) / std, 4)).Average();
            _kurtosisValues[j] = moment4 - 3; // Excess kurtosis (normal = 0)
        }

        // Select features based on preference
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferHighKurtosis)
        {
            // Prefer features with high absolute excess kurtosis
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => Math.Abs(_kurtosisValues[j]))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Prefer features with kurtosis close to normal
            _selectedIndices = Enumerable.Range(0, p)
                .OrderBy(j => Math.Abs(_kurtosisValues[j]))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KurtosisSelector has not been fitted.");

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
        throw new NotSupportedException("KurtosisSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KurtosisSelector has not been fitted.");

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
