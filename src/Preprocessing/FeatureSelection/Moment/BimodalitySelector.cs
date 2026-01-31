using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Bimodality Coefficient based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their bimodality coefficient, which indicates
/// whether a distribution has two distinct modes (peaks).
/// </para>
/// <para><b>For Beginners:</b> Bimodality measures if data has two separate groups.
/// A high bimodality coefficient (>0.555 for uniform threshold) suggests the feature
/// naturally separates data into two clusters, which can be useful for classification.
/// The formula is: BC = (skewness² + 1) / (kurtosis + 3 × (n-1)²/((n-2)(n-3)))
/// </para>
/// </remarks>
public class BimodalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighBimodality;

    private double[]? _bimodalityValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighBimodality => _preferHighBimodality;
    public double[]? BimodalityValues => _bimodalityValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BimodalitySelector(
        int nFeaturesToSelect = 10,
        bool preferHighBimodality = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighBimodality = preferHighBimodality;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 4)
            throw new ArgumentException("At least 4 samples are required for bimodality calculation.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _bimodalityValues = new double[p];

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
                _bimodalityValues[j] = 0;
                continue;
            }

            // Compute skewness
            double skewness = col.Select(v => Math.Pow((v - mean) / std, 3)).Average();

            // Compute excess kurtosis
            double kurtosis = col.Select(v => Math.Pow((v - mean) / std, 4)).Average() - 3;

            // Bimodality coefficient (Sarle's formula)
            // BC = (skewness² + 1) / (kurtosis + 3 × (n-1)²/((n-2)(n-3)))
            double skewSquared = skewness * skewness;
            double sampleCorrection = 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
            double denominator = kurtosis + sampleCorrection;

            // Avoid division by zero or negative
            if (denominator <= 0)
                denominator = 1e-10;

            _bimodalityValues[j] = (skewSquared + 1) / denominator;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferHighBimodality)
        {
            // Prefer features with high bimodality (likely two-group structure)
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _bimodalityValues[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Prefer features with low bimodality (more unimodal)
            _selectedIndices = Enumerable.Range(0, p)
                .OrderBy(j => _bimodalityValues[j])
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
            throw new InvalidOperationException("BimodalitySelector has not been fitted.");

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
        throw new NotSupportedException("BimodalitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BimodalitySelector has not been fitted.");

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
