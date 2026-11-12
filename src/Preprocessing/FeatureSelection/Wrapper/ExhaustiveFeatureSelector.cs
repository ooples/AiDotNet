using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Exhaustive Feature Selector that evaluates all possible feature subsets.
/// </summary>
/// <remarks>
/// <para>
/// The Exhaustive Feature Selector evaluates all possible combinations of features up to
/// a maximum subset size. This guarantees finding the optimal subset but has exponential
/// complexity O(2^p) where p is the number of features.
/// </para>
/// <para><b>For Beginners:</b> This is the brute-force approach: try every possible
/// combination of features and pick the best one. It's guaranteed to find the optimal
/// subset, but it's only practical for small numbers of features (typically under 20)
/// because the number of combinations grows exponentially.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ExhaustiveFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _minFeatures;
    private readonly int _maxFeatures;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int _nFolds;

    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MinFeatures => _minFeatures;
    public int MaxFeatures => _maxFeatures;
    public double BestScore => _bestScore;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ExhaustiveFeatureSelector(
        int minFeatures = 1,
        int maxFeatures = 10,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int nFolds = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minFeatures < 1)
            throw new ArgumentException("Minimum features must be at least 1.", nameof(minFeatures));
        if (maxFeatures < minFeatures)
            throw new ArgumentException("Maximum features must be >= minimum features.", nameof(maxFeatures));

        _minFeatures = minFeatures;
        _maxFeatures = maxFeatures;
        _scorer = scorer;
        _nFolds = nFolds;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExhaustiveFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int maxFeaturesActual = Math.Min(_maxFeatures, p);

        var scorer = _scorer ?? DefaultScorer;
        _bestScore = double.NegativeInfinity;
        _selectedIndices = null;

        // Generate and evaluate all combinations
        for (int k = _minFeatures; k <= maxFeaturesActual; k++)
        {
            foreach (var combination in GetCombinations(p, k))
            {
                double score = scorer(data, target, combination);
                if (score > _bestScore)
                {
                    _bestScore = score;
                    _selectedIndices = combination.OrderBy(x => x).ToArray();
                }
            }
        }

        // Fallback if nothing was selected
        if (_selectedIndices is null || _selectedIndices.Length == 0)
        {
            _selectedIndices = Enumerable.Range(0, Math.Min(_minFeatures, p)).ToArray();
        }

        IsFitted = true;
    }

    private static IEnumerable<int[]> GetCombinations(int n, int k)
    {
        if (k > n || k < 0)
            yield break;

        int[] result = new int[k];
        var stack = new Stack<int>();
        stack.Push(0);

        while (stack.Count > 0)
        {
            int index = stack.Count - 1;
            int value = stack.Pop();

            while (value < n)
            {
                result[index++] = value++;
                stack.Push(value);

                if (index == k)
                {
                    yield return (int[])result.Clone();
                    break;
                }
            }
        }
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

        // k-fold cross-validated R^2 of a least-squares linear regression on the subset, as mlxtend's
        // ExhaustiveFeatureSelector with scikit-learn's LinearRegression and KFold(n_splits=nFolds): contiguous
        // test folds, the first n % nFolds of them one row larger
        int n = data.Rows;
        var x = WrapperScoringOps.Columns(data, featureIndices);
        var y = Tensor<T>.FromVector(target);
        double totalScore = 0;
        int testStart = 0;

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int testSize = n / _nFolds + (fold < n % _nFolds ? 1 : 0);
            if (testSize == 0 || testSize == n)
                return double.NegativeInfinity;

            int testEnd = testStart + testSize;
            var slopes = WrapperScoringOps.LeastSquaresSlopes(
                WrapperScoringOps.ExcludeRows(x, testStart, testEnd), WrapperScoringOps.ExcludeRows(y, testStart, testEnd),
                out var xMean, out T yMean);
            var predictions = WrapperScoringOps.Predict(
                WrapperScoringOps.RowBlock(x, testStart, testSize), slopes, xMean, yMean);
            totalScore += WrapperScoringOps.RSquared(WrapperScoringOps.RowBlock(y, testStart, testSize), predictions);
            testStart = testEnd;
        }

        return totalScore / _nFolds;
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
            throw new InvalidOperationException("ExhaustiveFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ExhaustiveFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExhaustiveFeatureSelector has not been fitted.");

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
