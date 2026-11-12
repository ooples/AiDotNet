using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Exhaustive Search Feature Selection (for small feature sets).
/// </summary>
/// <remarks>
/// <para>
/// Exhaustive search evaluates all possible feature subsets to find the optimal
/// combination. This guarantees finding the best subset but is only feasible
/// for small numbers of features (typically less than 20).
/// </para>
/// <para><b>For Beginners:</b> This is like trying every possible combination
/// of ingredients to find the perfect recipe. It's guaranteed to find the best
/// answer, but becomes impractical with many features because the number of
/// combinations grows exponentially (2^n for n features).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ExhaustiveSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxFeaturesForExhaustive;

    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double BestScore => _bestScore;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ExhaustiveSearch(
        int nFeaturesToSelect = 5,
        int maxFeaturesForExhaustive = 15,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxFeaturesForExhaustive < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeaturesForExhaustive));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxFeaturesForExhaustive = maxFeaturesForExhaustive;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExhaustiveSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (p > _maxFeaturesForExhaustive)
            throw new InvalidOperationException(
                $"ExhaustiveSearch only supports up to {_maxFeaturesForExhaustive} features. " +
                $"Current data has {p} features. Use a different selection method.");

        int k = Math.Min(_nFeaturesToSelect, p);
        _bestScore = double.MinValue;
        _selectedIndices = null;

        // Generate all combinations of k features from p
        var indices = Enumerable.Range(0, p).ToArray();
        var combinations = GetCombinations(indices, k);

        foreach (var combo in combinations)
        {
            double score = EvaluateSubset(data, target, combo, n);
            if (score > _bestScore)
            {
                _bestScore = score;
                _selectedIndices = combo.ToArray();
            }
        }

        IsFitted = true;
    }

    private IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        if (k == 0)
        {
            yield return Array.Empty<int>();
            yield break;
        }

        if (elements.Length < k)
            yield break;

        for (int i = 0; i <= elements.Length - k; i++)
        {
            int first = elements[i];
            var rest = elements.Skip(i + 1).ToArray();

            foreach (var combo in GetCombinations(rest, k - 1))
            {
                var result = new int[k];
                result[0] = first;
                for (int j = 0; j < combo.Length; j++)
                    result[j + 1] = combo[j];
                yield return result;
            }
        }
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] featureIndices, int n)
    {
        // Average absolute correlation with target, penalized by the average absolute pairwise correlation
        // (redundancy) of the subset
        double relevance = featureIndices.Length > 0
            ? WrapperScoringOps.MeanAbsTargetCorrelation(data, target, featureIndices)
            : double.NaN;
        return relevance - 0.5 * WrapperScoringOps.MeanAbsPairCorrelation(data, featureIndices);
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
            throw new InvalidOperationException("ExhaustiveSearch has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ExhaustiveSearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExhaustiveSearch has not been fitted.");

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
