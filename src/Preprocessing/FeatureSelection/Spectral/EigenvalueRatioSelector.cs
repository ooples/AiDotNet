using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Spectral;

/// <summary>
/// Eigenvalue Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the ratio of top eigenvalues when features are
/// added, measuring how much each feature contributes to data variance.
/// </para>
/// <para><b>For Beginners:</b> Eigenvalues measure how much variance each
/// direction in data captures. This selector keeps features that contribute
/// most to the dominant directions of variance in your data.
/// </para>
/// </remarks>
public class EigenvalueRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _varianceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? VarianceContributions => _varianceContributions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EigenvalueRatioSelector(
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

        // Center data
        var means = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] -= means[j];

        // Compute variance contribution for each feature
        _varianceContributions = new double[p];
        double totalVariance = 0;

        for (int j = 0; j < p; j++)
        {
            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += X[i, j] * X[i, j];
            variance /= (n - 1);
            _varianceContributions[j] = variance;
            totalVariance += variance;
        }

        // Also consider covariance contribution
        for (int j = 0; j < p; j++)
        {
            double covSum = 0;
            for (int k = 0; k < p; k++)
            {
                if (k != j)
                {
                    double cov = 0;
                    for (int i = 0; i < n; i++)
                        cov += X[i, j] * X[i, k];
                    cov /= (n - 1);
                    covSum += Math.Abs(cov);
                }
            }
            // Features with unique information (low covariance) get bonus
            _varianceContributions[j] += _varianceContributions[j] / (1 + covSum / p);
        }

        // Normalize
        if (totalVariance > 0)
        {
            for (int j = 0; j < p; j++)
                _varianceContributions[j] /= totalVariance;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _varianceContributions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

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
            throw new InvalidOperationException("EigenvalueRatioSelector has not been fitted.");

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
        throw new NotSupportedException("EigenvalueRatioSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EigenvalueRatioSelector has not been fitted.");

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
