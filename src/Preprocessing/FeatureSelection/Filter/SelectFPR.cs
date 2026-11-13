using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Select features based on False Positive Rate threshold.
/// </summary>
/// <remarks>
/// <para>
/// Selects features with p-values below a specified alpha threshold. Does not control
/// for multiple testing, so at alpha=0.05, approximately 5% of null (non-informative)
/// features will be incorrectly selected.
/// </para>
/// <para><b>For Beginners:</b> This method uses statistical testing to find features
/// that are unlikely to be uninformative by chance. If you set alpha=0.05, it keeps
/// features where there's less than 5% chance the feature-target relationship is random.
/// However, when testing many features, some false positives slip through.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectFPR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _alpha;
    private readonly SelectKBestScoreFunc _scoreFunction;
    private readonly Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)>? _customScoreFunction;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Alpha => _alpha;
    /// <summary>
    /// Gets the built-in scoring function, used when no custom scorer was supplied.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunction => _scoreFunction;

    /// <summary>Gets the score of every feature after fitting.</summary>
    public Vector<T>? Scores => _scores;

    /// <summary>Gets the p-value of every feature after fitting.</summary>
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a selector that scores features with a built-in scikit-learn scoring function.
    /// </summary>
    /// <param name="alpha">The highest p-value of a kept feature (exclusive).</param>
    /// <param name="scoreFunction">
    /// The scoring function; defaults to <see cref="SelectKBestScoreFunc.FClassif"/> (scikit-learn's default); it must produce p-values.
    /// </param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectFPR(
        double alpha = 0.05,
        SelectKBestScoreFunc scoreFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        if (scoreFunction == SelectKBestScoreFunc.MutualInfoRegression)
            throw new ArgumentException(
                $"The {scoreFunction} scoring function produces no p-values; SelectFPR needs one that does.",
                nameof(scoreFunction));
        _alpha = alpha;
        _scoreFunction = scoreFunction;
    }

    /// <summary>
    /// Creates a selector with a custom scorer, like scikit-learn's <c>score_func</c>: it receives the data and the
    /// target and returns one score per feature (higher is better) and one p-value per feature.
    /// </summary>
    /// <param name="alpha">The highest p-value of a kept feature (exclusive).</param>
    /// <param name="scoreFunction">The scorer.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectFPR(
        double alpha,
        Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)> scoreFunction,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        if (scoreFunction is null)
            throw new ArgumentNullException(nameof(scoreFunction));

        _alpha = alpha;
        _scoreFunction = SelectKBestScoreFunc.FClassif;
        _customScoreFunction = scoreFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectFPR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

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

        // validated before any fitted state changes, so a failed refit keeps the previous fit whole
        if (pValues is null)
            throw new InvalidOperationException("SelectFPR needs a scoring function that returns p-values.");

        _nInputFeatures = p;
        _scores = scores;
        _pValues = pValues;

        // keep every feature whose p-value is below alpha (scikit-learn's SelectFpr); none passing selects none
        _selectedIndices = FilterStatisticTensorOps.MaskPositions(
            Engine.TensorLessThan(FilterStatisticTensorOps.AsTensor(pValues), NumOps.FromDouble(_alpha)));

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
            throw new InvalidOperationException("SelectFPR has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectFPR does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFPR has not been fitted.");

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
