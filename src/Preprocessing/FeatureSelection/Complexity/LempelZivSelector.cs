using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Lempel-Ziv Complexity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Lempel-Ziv complexity, which measures
/// the number of distinct patterns in a discretized sequence.
/// </para>
/// <para><b>For Beginners:</b> Lempel-Ziv complexity counts how many unique
/// patterns appear when reading a sequence. It's related to compression -
/// sequences that compress well have low complexity (predictable patterns),
/// while complex sequences have many unique patterns and don't compress well.
/// </para>
/// </remarks>
public class LempelZivSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _complexityValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? ComplexityValues => _complexityValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LempelZivSelector(
        int nFeaturesToSelect = 10,
        int nBins = 2,
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

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _complexityValues = new double[p];

        // Normalized complexity bound: n / log_k(n) where k = alphabet size
        double normFactor = n / (Math.Log(n) / Math.Log(_nBins));

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Discretize to symbols
            double min = col.Min();
            double max = col.Max();
            double range = max - min;

            var symbols = new int[n];
            if (range < 1e-10)
            {
                // Constant - all same symbol
                for (int i = 0; i < n; i++)
                    symbols[i] = 0;
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    int bin = (int)((col[i] - min) / range * (_nBins - 1));
                    symbols[i] = Math.Min(bin, _nBins - 1);
                }
            }

            // Compute Lempel-Ziv complexity
            int complexity = ComputeLZComplexity(symbols);

            // Normalize by theoretical bound
            _complexityValues[j] = normFactor > 0 ? complexity / normFactor : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _complexityValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int ComputeLZComplexity(int[] sequence)
    {
        int n = sequence.Length;
        if (n == 0) return 0;
        if (n == 1) return 1;

        var dictionary = new HashSet<string>();
        int complexity = 0;
        int i = 0;

        while (i < n)
        {
            int length = 1;
            string word = sequence[i].ToString();

            // Extend word while it exists in dictionary
            while (i + length <= n)
            {
                var sb = new System.Text.StringBuilder();
                for (int k = i; k < i + length; k++)
                    sb.Append(sequence[k]).Append(",");
                word = sb.ToString();

                if (!dictionary.Contains(word) || i + length == n)
                    break;

                length++;
            }

            dictionary.Add(word);
            complexity++;
            i += length;
        }

        return complexity;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LempelZivSelector has not been fitted.");

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
        throw new NotSupportedException("LempelZivSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LempelZivSelector has not been fitted.");

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
