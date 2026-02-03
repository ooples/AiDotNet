using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Gaussianity (Normality) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how closely their distributions match a Gaussian
/// (normal) distribution, using the Jarque-Bera test statistic.
/// </para>
/// <para><b>For Beginners:</b> Many statistical methods assume data follows a
/// bell curve (normal distribution). This selector measures how "normal" each
/// feature is. You can select features that are most normal (for parametric methods)
/// or least normal (to find interesting non-standard patterns).
/// </para>
/// </remarks>
public class GaussianitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferGaussian;

    private double[]? _jbStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferGaussian => _preferGaussian;
    public double[]? JBStatistics => _jbStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GaussianitySelector(
        int nFeaturesToSelect = 10,
        bool preferGaussian = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferGaussian = preferGaussian;
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

        _jbStatistics = new double[p];

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
                // Constant feature - infinite JB statistic (very non-normal)
                _jbStatistics[j] = double.MaxValue;
                continue;
            }

            // Compute skewness
            double skewness = col.Select(v => Math.Pow((v - mean) / std, 3)).Average();

            // Compute excess kurtosis
            double kurtosis = col.Select(v => Math.Pow((v - mean) / std, 4)).Average() - 3;

            // Jarque-Bera test statistic: JB = n/6 * (S² + K²/4)
            // Lower JB = more Gaussian
            _jbStatistics[j] = (n / 6.0) * (skewness * skewness + kurtosis * kurtosis / 4.0);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferGaussian)
        {
            // Prefer features with low JB statistic (more Gaussian)
            _selectedIndices = Enumerable.Range(0, p)
                .OrderBy(j => _jbStatistics[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Prefer features with high JB statistic (less Gaussian)
            _selectedIndices = Enumerable.Range(0, p)
                .Where(j => _jbStatistics[j] < double.MaxValue)
                .OrderByDescending(j => _jbStatistics[j])
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
            throw new InvalidOperationException("GaussianitySelector has not been fitted.");

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
        throw new NotSupportedException("GaussianitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GaussianitySelector has not been fitted.");

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
