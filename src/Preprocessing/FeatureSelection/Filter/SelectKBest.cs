using AiDotNet.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Selects the K best features based on a scoring function (scikit-learn's <c>SelectKBest</c>).
/// </summary>
/// <remarks>
/// <para>
/// A simple univariate feature selection that computes a score for each feature and selects the K highest
/// scoring features. As in scikit-learn the default scoring function is the ANOVA F-value for classification
/// (<see cref="SelectKBestScoreFunc.FClassif"/>, <c>f_classif</c>). For a continuous (regression) target pass
/// <see cref="SelectKBestScoreFunc.FRegression"/> (<c>f_regression</c>) or
/// <see cref="SelectKBestScoreFunc.MutualInfoRegression"/>, or supply any scorer of your own; the target is
/// never re-interpreted behind your back.
/// </para>
/// <para><b>For Beginners:</b> SelectKBest is the simplest approach - compute a score
/// for each feature independently and keep the top K. It's fast and works well when
/// features are truly independent, but may miss interactions between features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectKBest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _k;
    private readonly SelectKBestScoreFunc _scoreFunction;
    private readonly Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)>? _customScoreFunction;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>Gets the number of features to select.</summary>
    public int K => _k;

    /// <summary>
    /// Gets the built-in scoring function, used when no custom scorer was supplied.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunction => _scoreFunction;

    /// <summary>Gets the score of every feature after fitting.</summary>
    public Vector<T>? Scores => _scores;

    /// <summary>
    /// Gets the p-value of every feature after fitting, or null when the scoring function has none.
    /// </summary>
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a selector that scores features with a built-in scikit-learn scoring function.
    /// </summary>
    /// <param name="k">Number of features to select.</param>
    /// <param name="scoreFunction">
    /// The scoring function; defaults to <see cref="SelectKBestScoreFunc.FClassif"/> (scikit-learn's default).
    /// </param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectKBest(
        int k = 10,
        SelectKBestScoreFunc scoreFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
            throw new ArgumentException("K must be at least 1.", nameof(k));

        _k = k;
        _scoreFunction = scoreFunction;
    }

    /// <summary>
    /// Creates a selector with a custom scorer, like scikit-learn's <c>score_func</c>: it receives the data and the
    /// target and returns one score per feature (higher is better) and optionally one p-value per feature.
    /// </summary>
    /// <param name="k">Number of features to select.</param>
    /// <param name="scoreFunction">The scorer.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectKBest(
        int k,
        Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)> scoreFunction,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
            throw new ArgumentException("K must be at least 1.", nameof(k));
        if (scoreFunction is null)
            throw new ArgumentNullException(nameof(scoreFunction));

        _k = k;
        _scoreFunction = SelectKBestScoreFunc.FClassif;
        _customScoreFunction = scoreFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectKBest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // the K highest scores (NaN counts as the lowest score, equal scores go to the lower column)
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(FilterStatisticTensorOps.AsTensor(scores), Math.Min(_k, p));

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
            throw new InvalidOperationException("SelectKBest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectKBest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectKBest has not been fitted.");

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
