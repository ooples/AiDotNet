using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Approximate Entropy (ApEn) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their approximate entropy, which quantifies
/// the unpredictability and complexity of time series data.
/// </para>
/// <para><b>For Beginners:</b> Approximate entropy measures how likely it is
/// that similar patterns in a sequence remain similar when extended by one point.
/// Low ApEn means the sequence is predictable (patterns repeat); high ApEn means
/// it's complex and hard to predict. Often used for physiological signals.
/// </para>
/// </remarks>
public class ApproximateEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly double _tolerance;

    private double[]? _apenValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public double Tolerance => _tolerance;
    public double[]? ApEnValues => _apenValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ApproximateEntropySelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 2,
        double tolerance = 0.2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 1)
            throw new ArgumentException("Embedding dimension must be at least 1.", nameof(embeddingDimension));
        if (tolerance <= 0)
            throw new ArgumentException("Tolerance must be positive.", nameof(tolerance));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < _embeddingDimension + 2)
            throw new ArgumentException($"Need at least {_embeddingDimension + 2} samples.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _apenValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Calculate standard deviation for tolerance scaling
            double mean = col.Average();
            double std = Math.Sqrt(col.Select(v => (v - mean) * (v - mean)).Average());
            double r = _tolerance * std;

            if (r < 1e-10)
            {
                _apenValues[j] = 0;
                continue;
            }

            // Compute phi for m and m+1
            double phiM = ComputePhi(col, _embeddingDimension, r);
            double phiM1 = ComputePhi(col, _embeddingDimension + 1, r);

            // ApEn = phi(m) - phi(m+1)
            _apenValues[j] = phiM - phiM1;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _apenValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePhi(double[] data, int m, double r)
    {
        int n = data.Length;
        int count = n - m + 1;
        if (count <= 0) return 0;

        var logs = new double[count];

        for (int i = 0; i < count; i++)
        {
            int matches = 0;
            for (int k = 0; k < count; k++)
            {
                // Check if patterns match within tolerance
                bool isMatch = true;
                for (int l = 0; l < m; l++)
                {
                    if (Math.Abs(data[i + l] - data[k + l]) > r)
                    {
                        isMatch = false;
                        break;
                    }
                }
                if (isMatch) matches++;
            }
            logs[i] = Math.Log((double)matches / count);
        }

        return logs.Average();
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ApproximateEntropySelector has not been fitted.");

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
        throw new NotSupportedException("ApproximateEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ApproximateEntropySelector has not been fitted.");

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
