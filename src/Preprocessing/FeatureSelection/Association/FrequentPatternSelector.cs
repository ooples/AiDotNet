using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Association;

/// <summary>
/// Frequent Pattern-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Mines frequent patterns in feature-target associations and selects
/// features that appear in the most discriminative frequent patterns.
/// </para>
/// <para><b>For Beginners:</b> Frequent patterns are combinations of features
/// that appear together often. This method finds which individual features
/// appear most often in patterns that are predictive of the target class,
/// and selects those features.
/// </para>
/// </remarks>
public class FrequentPatternSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSupport;
    private readonly int _maxPatternSize;

    private double[]? _patternScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? PatternScores => _patternScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FrequentPatternSelector(
        int nFeaturesToSelect = 10,
        double minSupport = 0.1,
        int maxPatternSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minSupport = minSupport;
        _maxPatternSize = maxPatternSize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FrequentPatternSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            double yVal = NumOps.ToDouble(target[i]);
            y[i] = yVal > 0.5 ? 1 : 0;
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Binarize features using median split
        var binarized = new bool[n, p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X[i, j];
            Array.Sort(col);
            double median = col[n / 2];
            for (int i = 0; i < n; i++)
                binarized[i, j] = X[i, j] >= median;
        }

        // Find frequent 1-itemsets
        var freq1 = new Dictionary<int, int>();
        for (int j = 0; j < p; j++)
        {
            int count = 0;
            for (int i = 0; i < n; i++)
                if (binarized[i, j] && y[i] == 1)
                    count++;
            if ((double)count / n >= _minSupport)
                freq1[j] = count;
        }

        // Score features based on frequent itemsets they participate in
        _patternScores = new double[p];
        foreach (var kvp in freq1)
            _patternScores[kvp.Key] += (double)kvp.Value / n;

        // Find frequent 2-itemsets and higher (up to maxPatternSize)
        if (_maxPatternSize >= 2)
        {
            var freq1Keys = freq1.Keys.ToList();
            for (int size = 2; size <= Math.Min(_maxPatternSize, freq1Keys.Count); size++)
            {
                foreach (var pattern in GetCombinations(freq1Keys, size))
                {
                    int count = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if (y[i] != 1) continue;
                        bool allTrue = pattern.All(j => binarized[i, j]);
                        if (allTrue) count++;
                    }

                    if ((double)count / n >= _minSupport)
                    {
                        double patternScore = (double)count / n / size;
                        foreach (int j in pattern)
                            _patternScores[j] += patternScore;
                    }
                }
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _patternScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private IEnumerable<List<int>> GetCombinations(List<int> items, int size)
    {
        if (size == 0)
        {
            yield return new List<int>();
            yield break;
        }
        if (items.Count < size) yield break;

        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetCombinations(items.Skip(i + 1).ToList(), size - 1))
            {
                var combo = new List<int> { items[i] };
                combo.AddRange(rest);
                yield return combo;
            }
        }
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FrequentPatternSelector has not been fitted.");

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
        throw new NotSupportedException("FrequentPatternSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FrequentPatternSelector has not been fitted.");

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
