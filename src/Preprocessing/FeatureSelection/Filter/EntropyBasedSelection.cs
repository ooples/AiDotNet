using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Entropy-Based Feature Selection using discretized feature entropy.
/// </summary>
/// <remarks>
/// <para>
/// Measures the information content of each feature using Shannon entropy.
/// Features with higher entropy contain more information and discriminative power.
/// Features with very low entropy (constant or near-constant) are removed.
/// </para>
/// <para><b>For Beginners:</b> Entropy measures how "mixed up" or unpredictable
/// a feature's values are. A feature that's always the same value has zero
/// entropy and tells you nothing useful. Features with higher entropy have
/// more variety and can better distinguish between different samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EntropyBasedSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _minEntropyRatio;

    private double[]? _entropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? EntropyScores => _entropyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EntropyBasedSelection(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double minEntropyRatio = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _minEntropyRatio = minEntropyRatio;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _entropyScores = new double[p];
        double maxPossibleEntropy = Math.Log(_nBins);

        for (int j = 0; j < p; j++)
        {
            // Get feature values
            var values = new double[n];
            double minVal = double.MaxValue;
            double maxVal = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, values[i]);
                maxVal = Math.Max(maxVal, values[i]);
            }

            // Discretize into bins
            var binCounts = new int[_nBins];
            double range = maxVal - minVal;

            if (range < 1e-10)
            {
                // Constant feature
                _entropyScores[j] = 0;
                continue;
            }

            for (int i = 0; i < n; i++)
            {
                int bin = (int)((values[i] - minVal) / range * (_nBins - 1));
                bin = Math.Min(bin, _nBins - 1);
                binCounts[bin]++;
            }

            // Compute entropy
            double entropy = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] > 0)
                {
                    double prob = (double)binCounts[b] / n;
                    entropy -= prob * Math.Log(prob);
                }
            }

            _entropyScores[j] = entropy;
        }

        // Filter by minimum entropy ratio
        var candidates = Enumerable.Range(0, p)
            .Where(j => _entropyScores[j] >= _minEntropyRatio * maxPossibleEntropy)
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        _selectedIndices = candidates
            .OrderByDescending(j => _entropyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EntropyBasedSelection has not been fitted.");

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
        throw new NotSupportedException("EntropyBasedSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EntropyBasedSelection has not been fitted.");

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
