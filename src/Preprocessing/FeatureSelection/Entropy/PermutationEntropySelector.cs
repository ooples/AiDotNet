using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Permutation Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on permutation entropy, which measures the complexity
/// of time series or sequential data based on ordinal patterns.
/// </para>
/// <para><b>For Beginners:</b> Permutation entropy looks at the order of values
/// in small windows. It counts how many different orderings appear and how often.
/// High permutation entropy means complex, unpredictable sequences; low entropy
/// means regular, predictable patterns like trends.
/// </para>
/// </remarks>
public class PermutationEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly int _delay;

    private double[]? _entropyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public int Delay => _delay;
    public double[]? EntropyValues => _entropyValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PermutationEntropySelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 3,
        int delay = 1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 2)
            throw new ArgumentException("Embedding dimension must be at least 2.", nameof(embeddingDimension));
        if (delay < 1)
            throw new ArgumentException("Delay must be at least 1.", nameof(delay));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _delay = delay;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        int minLength = (_embeddingDimension - 1) * _delay + 1;
        if (n < minLength)
            throw new ArgumentException($"Need at least {minLength} samples for embedding dimension {_embeddingDimension} and delay {_delay}.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _entropyValues = new double[p];

        // Maximum entropy for m! possible permutations
        int factorial = 1;
        for (int i = 2; i <= _embeddingDimension; i++)
            factorial *= i;
        double maxEntropy = Math.Log(factorial) / Math.Log(2);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Count permutation patterns
            var patternCounts = new Dictionary<string, int>();
            int totalPatterns = n - (_embeddingDimension - 1) * _delay;

            for (int i = 0; i <= n - minLength; i++)
            {
                // Extract embedded vector
                var embedded = new double[_embeddingDimension];
                for (int d = 0; d < _embeddingDimension; d++)
                    embedded[d] = col[i + d * _delay];

                // Get permutation pattern (argsort)
                var indices = Enumerable.Range(0, _embeddingDimension)
                    .OrderBy(idx => embedded[idx])
                    .ToArray();
                string pattern = string.Join(",", indices);

                if (!patternCounts.ContainsKey(pattern))
                    patternCounts[pattern] = 0;
                patternCounts[pattern]++;
            }

            // Compute entropy
            double entropy = 0;
            foreach (var count in patternCounts.Values)
            {
                double prob = (double)count / totalPatterns;
                if (prob > 0)
                    entropy -= prob * Math.Log(prob) / Math.Log(2);
            }

            // Normalize by maximum entropy
            _entropyValues[j] = maxEntropy > 0 ? entropy / maxEntropy : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _entropyValues[j])
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
            throw new InvalidOperationException("PermutationEntropySelector has not been fitted.");

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
        throw new NotSupportedException("PermutationEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationEntropySelector has not been fitted.");

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
