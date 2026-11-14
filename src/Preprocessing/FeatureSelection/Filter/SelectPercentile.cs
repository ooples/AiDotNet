using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Selects features based on a percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// Similar to SelectKBest but selects a percentage of features rather than a fixed
/// number. Useful when you want a consistent selection ratio across datasets of
/// different sizes.
/// </para>
/// <para><b>For Beginners:</b> Instead of saying "keep the top 10 features," you say
/// "keep the top 25% of features." This is useful when you want a consistent reduction
/// ratio that scales with the number of features in your dataset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _percentile;
    private readonly SelectKBestScoreFunc _scoreFunction;
    private readonly Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)>? _customScoreFunction;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Percentile => _percentile;
    /// <summary>
    /// Gets the built-in scoring function, used when no custom scorer was supplied.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunction => _scoreFunction;

    /// <summary>Gets the score of every feature after fitting.</summary>
    public Vector<T>? Scores => _scores;

    /// <summary>Gets the p-value of every feature after fitting, or null when the scoring function has none.</summary>
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a selector that scores features with a built-in scikit-learn scoring function.
    /// </summary>
    /// <param name="percentile">Percentage of features to keep (scikit-learn's default is 10).</param>
    /// <param name="scoreFunction">
    /// The scoring function; defaults to <see cref="SelectKBestScoreFunc.FClassif"/> (scikit-learn's default).
    /// </param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectPercentile(
        double percentile = 10,
        SelectKBestScoreFunc scoreFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));
        _percentile = percentile;
        _scoreFunction = scoreFunction;
    }

    /// <summary>
    /// Creates a selector with a custom scorer, like scikit-learn's <c>score_func</c>: it receives the data and the
    /// target and returns one score per feature (higher is better) and optionally one p-value per feature.
    /// </summary>
    /// <param name="percentile">Percentage of features to keep (scikit-learn's default is 10).</param>
    /// <param name="scoreFunction">The scorer.</param>
    public SelectPercentile(
        double percentile,
        Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)> scoreFunction)
        : base(null)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));
        if (scoreFunction is null)
            throw new ArgumentNullException(nameof(scoreFunction));

        _percentile = percentile;
        _scoreFunction = SelectKBestScoreFunc.FClassif;
        _customScoreFunction = scoreFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectPercentile requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        Vector<T> scores;
        Vector<T>? pValues;
        if (_customScoreFunction is not null)
        {
            (scores, pValues) = _customScoreFunction(data, target);
            if (scores is null || scores.Length != p)
                throw new InvalidOperationException("The scoring function must return one score per feature.");
            if (pValues is not null && pValues.Length != p)
                throw new InvalidOperationException("The scoring function must return one p-value per feature.");
        }
        else
        {
            (scores, pValues) = StatisticalTestHelper.ComputeScores(data, target, _scoreFunction);
        }

        _scores = scores;
        _pValues = pValues;

        // scikit-learn's percentile rule: scores above the (100 - percentile)-th percentile, then ties at it
        _selectedIndices = FeatureScoringTensorOps.PercentileSupport(FilterStatisticTensorOps.AsTensor(scores), _percentile);

        IsFitted = true;
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
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectPercentile does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
