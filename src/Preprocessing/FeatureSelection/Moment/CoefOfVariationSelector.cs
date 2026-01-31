using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Coefficient of Variation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their coefficient of variation (CV), which measures
/// the relative variability compared to the mean.
/// </para>
/// <para><b>For Beginners:</b> CV is the ratio of standard deviation to the mean.
/// Features with higher CV have more relative variability, which often indicates
/// they carry useful information for distinguishing between samples.
/// </para>
/// </remarks>
public class CoefOfVariationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighCV;

    private double[]? _cvValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighCV => _preferHighCV;
    public double[]? CVValues => _cvValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CoefOfVariationSelector(
        int nFeaturesToSelect = 10,
        bool preferHighCV = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighCV = preferHighCV;
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

        _cvValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double std = Math.Sqrt(col.Select(v => (v - mean) * (v - mean)).Average());

            // CV = std / |mean| (use absolute mean to handle negative values)
            _cvValues[j] = Math.Abs(mean) > 1e-10 ? std / Math.Abs(mean) : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferHighCV)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _cvValues[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = Enumerable.Range(0, p)
                .Where(j => _cvValues[j] > 1e-10)
                .OrderBy(j => _cvValues[j])
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
            throw new InvalidOperationException("CoefOfVariationSelector has not been fitted.");

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
        throw new NotSupportedException("CoefOfVariationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CoefOfVariationSelector has not been fitted.");

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
