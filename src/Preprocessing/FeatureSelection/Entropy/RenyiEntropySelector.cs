using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Renyi Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Renyi entropy, which generalizes Shannon entropy
/// with a tunable parameter alpha.
/// </para>
/// <para><b>For Beginners:</b> Renyi entropy is a family of entropy measures. With
/// alpha=1, it equals Shannon entropy. Different alpha values emphasize different
/// aspects of the distribution - low alpha focuses on rare events, high alpha on
/// common events.
/// </para>
/// </remarks>
public class RenyiEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _alpha;

    private double[]? _renyiEntropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double Alpha => _alpha;
    public double[]? RenyiEntropies => _renyiEntropies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RenyiEntropySelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double alpha = 2.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0 || Math.Abs(alpha - 1) < 1e-10)
            throw new ArgumentException("Alpha must be positive and not equal to 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _alpha = alpha;
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

        _renyiEntropies = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Discretize
            double xMin = col.Min();
            double xMax = col.Max();
            double xRange = xMax - xMin;
            var bins = new int[_nBins];

            for (int i = 0; i < n; i++)
            {
                int binIdx = xRange > 1e-10
                    ? Math.Min(_nBins - 1, (int)((col[i] - xMin) / xRange * _nBins))
                    : 0;
                bins[binIdx]++;
            }

            // Compute Renyi entropy: H_alpha(X) = 1/(1-alpha) * log(sum(p^alpha))
            double sumPowAlpha = 0;
            for (int b = 0; b < _nBins; b++)
            {
                double p_b = (double)bins[b] / n;
                if (p_b > 1e-10)
                    sumPowAlpha += Math.Pow(p_b, _alpha);
            }

            _renyiEntropies[j] = sumPowAlpha > 1e-10
                ? Math.Log(sumPowAlpha) / Math.Log(2) / (1 - _alpha)
                : 0;
        }

        // Select features with highest entropy (most informative)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _renyiEntropies[j])
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
            throw new InvalidOperationException("RenyiEntropySelector has not been fitted.");

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
        throw new NotSupportedException("RenyiEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RenyiEntropySelector has not been fitted.");

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
