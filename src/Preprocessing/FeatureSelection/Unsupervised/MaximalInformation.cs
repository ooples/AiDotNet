using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Maximal Information feature selection based on entropy for unsupervised learning.
/// </summary>
/// <remarks>
/// <para>
/// Maximal Information selects features with highest entropy (information content).
/// Features with higher entropy contain more information and can better capture
/// the underlying structure of the data.
/// </para>
/// <para><b>For Beginners:</b> Entropy measures how "surprising" or "uncertain"
/// a feature's values are. A feature that's always the same is boring (low entropy).
/// A feature with lots of different values is informative (high entropy). This method
/// selects the most informative features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MaximalInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _entropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? Entropies => _entropies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MaximalInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _entropies = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get feature values
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            // Discretize and count
            var binCounts = new int[_nBins];
            double range = maxVal - minVal;

            if (range < 1e-10)
            {
                // All values are the same - zero entropy
                _entropies[j] = 0;
                continue;
            }

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = (int)((val - minVal) / range * (_nBins - 1));
                bin = Math.Min(_nBins - 1, Math.Max(0, bin));
                binCounts[bin]++;
            }

            // Compute entropy
            double entropy = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] > 0)
                {
                    double p_b = (double)binCounts[b] / n;
                    entropy -= p_b * Math.Log(p_b);
                }
            }

            _entropies[j] = entropy;
        }

        // Select top features by entropy
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _entropies
            .Select((e, idx) => (Entropy: e, Index: idx))
            .OrderByDescending(x => x.Entropy)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformation has not been fitted.");

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
        throw new NotSupportedException("MaximalInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformation has not been fitted.");

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
