using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FrequentPatternSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSupport;
    private readonly int _maxPatternSize;

    private Vector<T>? _patternScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? PatternScores => _patternScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FrequentPatternSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var positive = engine.Reshape(
            engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)), new[] { n, 1 });

        // Binarize features using a median split (upper median: the element at n / 2 of the sorted column),
        // keeping only the positive transactions.
        var median = PreprocessingTensorOps.ColumnOrderStatistic(X, n / 2);
        var positiveItems = engine.TensorMultiply(PreprocessingTensorOps.AtLeastRow(X, median), positive);

        // Frequent 1-itemsets
        T count = NumOps.FromDouble(n);
        // Counted as int: a sum of 0/1 entries in T stops counting at 2048 for Half (2048 + 1 rounds back to 2048)
        var support1 = engine.TensorDivideScalar(
            engine.ReduceSum(positiveItems.Cast<int>(), new[] { 0 }, false).Cast<T>(), count);
        var frequent1 = PreprocessingTensorOps.AtLeast(support1, NumOps.FromDouble(_minSupport));
        var freq1Keys = PreprocessingTensorOps.MaskPositions(frequent1).ToList();

        // Score features based on the frequent itemsets they participate in
        var scores = engine.TensorMultiply(support1, frequent1);

        // Frequent 2-itemsets and higher (up to maxPatternSize): per size, the summed support counts
        // of the frequent patterns each feature belongs to.
        int maxSize = Math.Min(_maxPatternSize, freq1Keys.Count);
        if (_maxPatternSize >= 2 && maxSize >= 2)
        {
            int sizes = maxSize - 1;
            var countSums = new int[sizes * p];
            var missingItems = PreprocessingTensorOps.Not(positiveItems);
            T zero = NumOps.Zero;
            for (int size = 2; size <= maxSize; size++)
            {
                foreach (var pattern in GetCombinations(freq1Keys, size))
                {
                    // A transaction supports the pattern when none of its items is missing.
                    var columns = new Tensor<int>(pattern.ToArray(), new[] { pattern.Count });
                    var missingPerRow = engine.ReduceSum(engine.TensorIndexSelect(missingItems, columns, 1), new[] { 1 }, false);
                    int patternCount = engine.TensorCountNonzero(engine.TensorEquals(missingPerRow, zero));

                    if ((double)patternCount / n >= _minSupport)
                    {
                        foreach (int j in pattern)
                            countSums[(size - 2) * p + j] += patternCount;
                    }
                }
            }

            // score += count / n / size for every frequent pattern. The ratio is formed from the int counts: the
            // summed counts and n * size overflow T types such as Half (max 65504), giving inf / inf = NaN.
            var sizeScores = new Tensor<T>(Enumerable.Range(0, sizes * p)
                .Select(k => NumOps.FromDouble((double)countSums[k] / ((double)n * (k / p + 2))))
                .ToArray(), new[] { sizes, p });
            scores = engine.TensorAdd(scores, engine.ReduceSum(sizeScores, new[] { 0 }, false));
        }

        _patternScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FrequentPatternSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FrequentPatternSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FrequentPatternSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
